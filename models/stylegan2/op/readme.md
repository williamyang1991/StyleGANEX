Code from [rosinality-stylegan2-pytorch-cp](https://github.com/senior-sigan/rosinality-stylegan2-pytorch-cpu)

Scripts to convert rosinality/stylegan2-pytorch to the CPU compatible format

If you would like to use CPU for testing or have a problem regarding the cpp extention (fused and upfirdn2d), please make the following changes:

Change `models.stylegan2.op_old`  to `models.stylegan2.op`

https://github.com/williamyang1991/StyleGANEX/blob/73b580cc7eb757e36701c094456e9ee02078d03e/models/stylegan2/model.py#L8
