import numpy as np

class ConfMatrix:
    def __init__(self, true_labels, preds, labels=None):
        true_labels = np.array(true_labels)
        preds = np.array(preds)
        confmat = []
        self.labels = list(set(true_labels) | set(preds))
        if not labels:
            self.print_labels = self.labels[:]
        else:
            self.print_labels = list(labels)[:]
        for true in self.labels:
            sublist = []
            for predicted in self.labels:
                sublist.append(((true_labels == true) & (preds == predicted)).sum())
            confmat.append(sublist)
        self.confmat = np.array(confmat)

    def __str__(self):
        block = ["|".join(["{:8.8}".format(str(x)) for x in [head] + list(line)]) for head, line in zip([""] + self.print_labels, [self.print_labels] + list(self.confmat))]
        height = len(block)
        width = max([len(x) for x in block])
        if height > 4:
            a = [("{:^" + str(width) + "}").format("PREDICTED")] + ["{} {}".format(letter, line) for letter, line in zip(("{:^" + str(height + 1) + "}").format("TRUE"), block)]
            return "\n".join(a)
        else:
            return "\n".join([("{:^" + str(width) + "}").format("PREDICTED")] + ["{}{}".format(letter, line) for letter, line in zip(["     ", "TRUE "] + ["     "] * 2, block)])

    def __repr__(self):
        return self.__str__()

    def f2(self):
        # the variables may have the opposite meaning, relation still holds
        prec = np.diag(self.confmat) / self.confmat.sum(1)
        rec = np.diag(self.confmat) / self.confmat.sum(0)
        return 2 * prec * rec / (prec + rec)
