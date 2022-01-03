from prettytable import PrettyTable


def collect_model_info(model, rich_text=False):
    def bool2str(input):
        if input:
            return "Y"
        else:
            return "N"

    def shape_str(size):
        size = [str(s) for s in size]
        return "X".join(size)

    def min_max_str(input):
        return "Min:{:.3f} Max:{:.3f}".format(input.min(), input.max())

    def param_size(size, dtype="float32"):
        if dtype == "float32":
            size = size * 4
        elif dtype == "float16":
            size = size * 2
        else:
            raise NotImplementedError()
        return size / 1024.0 / 1024.0

    if not rich_text:
        table = PrettyTable(["Parameter Name", "Requires Grad", "Shape", "Value Scale"])
        total_size = 0
        for name, param in model.named_parameters():
            total_size += param.numel()
            table.add_row(
                [
                    name,
                    bool2str(param.requires_grad),
                    shape_str(param.size()),
                    min_max_str(param),
                ]
            )
        table.add_row(
            ["Total Number Of Params", "fp32", "%.4f" % param_size(total_size), "MB"]
        )
        return "\n" + table.get_string(title="Model Information")
    else:
        pass
