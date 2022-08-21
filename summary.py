from nets.yolo import Yolov6


if __name__ == '__main__':
    inputs = [640, 640, 3]
    phi = 'n'
    model = Yolov6(inputs=inputs, phi=phi)
    model.summary()
    print(model.output)
