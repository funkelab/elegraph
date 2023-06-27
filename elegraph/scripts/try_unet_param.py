def unet(input, num_conv, levels, downsample_factor=2):
    count = input
    # print(count)
    # downsample
    for i in range(levels):
        for j in range(num_conv):
            count -= 2  # from 3x3x3 kernel
            # print(count)
        if i != levels - 1:
            if count % downsample_factor != 0:
                return "not viable input"
            count = count // downsample_factor

    # upsample
    for i in range(levels - 1):
        count = count * downsample_factor
        # print(count)
        for j in range(num_conv):
            count -= 2
            # print(count)
    if count < 0:
        return "not viable input"
    return count


for num in range(120, 200):
    for i in range(4, 7):
        if unet(num, 2, i) != "not viable input":
            print("input: ", num, "levels: ", i, "output: ", unet(num, 2, i))
