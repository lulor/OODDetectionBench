
def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params


class LogUnbuffered:

    def __init__(self, args, stream, out_file=None):
        self.args = args
        if self.args.distributed and self.args.global_rank > 0:
            self.out_file = None
            return

        self.stream = stream
        if out_file:
            self.out_file = open(out_file, 'a')
        else:
            self.out_file = None

    def write(self, data):
        if self.args.distributed and self.args.global_rank > 0:
            return
        self.stream.write(data)
        if self.out_file:
            self.out_file.write(data)    # Write the data of stdout here to a text file as well
        self.flush()
    
    def flush(self):
        if self.args.distributed and self.args.global_rank > 0:
            return
        self.stream.flush()
        if self.out_file:
            self.out_file.flush()

    def close(self):
        if self.out_file:
            self.out_file.close()

def write_preds(filename, cs_preds, normality_scores):
    if cs_preds is not None and normality_scores is not None:
        assert len(cs_preds) == len(normality_scores), "Size mismatch"
    with open(filename, "w") as f:
        if cs_preds is not None and normality_scores is not None:
            for id_pred, ood_score in zip(cs_preds, normality_scores):
                f.write(f"{id_pred},{ood_score}\n")
        else:
            scores = cs_preds if normality_scores is None else normality_scores
            for s in scores:
                f.write(s + "\n")
