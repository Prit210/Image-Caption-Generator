import torch

def beam_search(model, enc, beam=16, max_len=30):
    device = enc.device
    seqs = [([1], 0.0)]

    for _ in range(max_len):
        all_candidates = []

        for seq, score in seqs:
            if seq[-1] == 2:
                all_candidates.append((seq, score))
                continue

            cap = torch.tensor(seq, device=device).unsqueeze(0)
            out = model.dec(enc, cap)
            probs = out[0, -1].softmax(-1)
            topk = probs.topk(beam)

            for i in range(beam):
                token = topk.indices[i].item()
                prob = topk.values[i].item()
                candidate = (seq + [token], score - torch.log(torch.tensor(prob)).item())
                all_candidates.append(candidate)

        seqs = sorted(all_candidates, key=lambda x: x[1])[:beam]

        if all(s[0][-1] == 2 for s in seqs):
            break

    return seqs[0][0]