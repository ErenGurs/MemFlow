import torch


class ModuleBackwarp(torch.nn.Module):
    def __init__(self):
        super(ModuleBackwarp, self).__init__()
    # end

    def forward(self, tensor_input, tensor_flow):
        if (not hasattr(self, 'tensorGrid') or (
                self.tensorGrid.size(2) != tensor_flow.size(2)) or (
                self.tensorGrid.size(3) != tensor_flow.size(3))):
            tensor_horizontal = torch.linspace(
                -1.0, 1.0, tensor_flow.size(3)
            ).view(1, 1, 1, tensor_flow.size(3)).expand(tensor_flow.size(0), -1, tensor_flow.size(2), -1)
            tensor_vertical = torch.linspace(
                -1.0, 1.0, tensor_flow.size(2)
            ).view(1, 1, tensor_flow.size(2), 1).expand(tensor_flow.size(0), -1, -1, tensor_flow.size(3))

            self.tensorGrid = torch.cat([tensor_horizontal, tensor_vertical], 1)
        # end
        return torch.nn.functional.grid_sample(
            input=tensor_input,
            grid=(self.tensorGrid.type_as(tensor_input) + torch.cat(
                [tensor_flow[:, 0:1, :, :] / ((tensor_input.size(3) - 1.0) / 2.0),
                    tensor_flow[:, 1:2, :, :] / ((tensor_input.size(2) - 1.0) / 2.0)], 1)
            ).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
    # end
# end


cache_backwarp = {}


def function_backwarp(tensor_input, tensor_flow):
    str_cache = tensor_input.type() + ' - ' + str(tensor_flow.size())

    if str_cache not in cache_backwarp:
        cache_backwarp[str_cache] = ModuleBackwarp()

        if tensor_input.is_cuda:
            cache_backwarp[str_cache] = cache_backwarp[str_cache].cuda()
        # end
    # end

    return cache_backwarp[str_cache](tensor_input, tensor_flow)
# end
