import cv2


def count():
    def fun(bbox):
        return bbox.shape[-2]

    return fun


def head_cropping(width_object=80, out_size=64):
    size = (out_size, out_size)
    resize_img = lambda x: cv2.resize(x, size) / 255

    def fun(bbox, img):
        heads = []

        for b in bbox:
            center = (b[3] + b[1]) // 2
            top = 0 if center < width_object else center - width_object
            bot = center + width_object

            center = (b[2] + b[0]) // 2
            left = 0 if center < width_object else center - width_object
            right = center + width_object

            heads.append(
                resize_img(img[top: bot, left: right])
            )

        return heads

    return fun
