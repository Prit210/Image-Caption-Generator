
import torch


def beam_search(model, enc, beam=16, max_len=30, alpha=0.65, block_ngram=3):
    """
    Beam search with:
      - Length penalty (Wu et al.)
      - n-gram blocking (no repeated n-grams of size block_ngram)
      - Diversity bonus
    enc   : (1, HW, dim) on correct device
    model : unwrapped model
    """
    device = enc.device
    seqs   = [([1], 0.0)]

    def _blocked(seq, token, n):
        """Return True if adding token creates a repeated n-gram."""
        if len(seq) < n - 1:
            return False
        candidate = seq[-(n-1):] + [token]
        for i in range(len(seq) - n + 1):
            if seq[i:i+n] == candidate:
                return True
        return False

    def _score(seq, raw):
        lp      = ((5 + len(seq)) / 6) ** alpha
        sort_sc = raw / lp
        content = [t for t in seq if t not in (0, 1, 2, 3)]
        if content:
            sort_sc += 0.15 * len(set(content)) / len(content)
        return sort_sc

    for _ in range(max_len):
        candidates = []
        for seq, raw_score in seqs:
            if seq[-1] == 2:
                candidates.append((seq, raw_score, _score(seq, raw_score)))
                continue
            cap   = torch.tensor(seq, device=device).unsqueeze(0)
            out   = model.dec(enc, cap)
            probs = out[0, -1].float().softmax(-1)
            top   = probs.topk(beam * 2)

            added = 0
            for i in range(beam * 2):
                if added >= beam:
                    break
                token = top.indices[i].item()
                if block_ngram and _blocked(seq, token, block_ngram):
                    continue
                lp      = top.values[i].clamp(min=1e-9).log().item()
                new_seq = seq + [token]
                new_raw = raw_score + lp
                candidates.append((new_seq, new_raw, _score(new_seq, new_raw)))
                added += 1

        candidates.sort(key=lambda x: x[2], reverse=True)
        seqs = [(s, r) for s, r, _ in candidates[:beam]]
        if not seqs:
            break  # all candidates blocked — return best so far
        if all(s[-1] == 2 for s, _ in seqs):
            break

    # Guard: if seqs is somehow empty, return just <start>
    return seqs[0][0] if seqs else [1, 2]
