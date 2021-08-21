from cascade import *
from colab_function import *


class BusTrackerCascade(Cascade):

    def __init__(self, weight, lifetime=50, min_iou=.2, max_gap=80, min_sim=.5):

        self.lifetime = lifetime
        self.min_iou = min_iou
        self.max_gap = max_gap
        self.min_sim = min_sim

        self.head_cropping = CascadeFunction(
            head_cropping(),
            name="Разрезалка"
        )
        self.sim_check = CascadeModel(weight)

        self.name = "Автобусный трекер Каскад"

        self.track = {}

        self.bboxes = []

    @staticmethod
    def bboxes_iou(boxes1, boxes2):
        """Get IoU between two boxes

        Parameters:
        boxes1 (np.array): four coordinates first box

        boxes2 (np.array): four coordinates second box

      """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        return ious

    @staticmethod
    def getDistance(box1, box2):
        """Get distance between two boxes

        Parameters:
        box1 (np.array): first box

        box2 (np.array): first box
      """
        centerBBox = lambda box: ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        center1 = centerBBox(box1)
        center2 = centerBBox(box2)
        return np.sqrt(
            np.square(center2[0] - center1[0]) +
            np.square(center2[1] - center1[1])
        )

    def sorted_iou(self, bbox):
        iou_list = []

        for i in self.track:
            iou = self.bboxes_iou(self.track[i][1], bbox)
            if iou > self.min_iou:
                iou_list.append((i, iou))

        iou_list = sorted(iou_list, key=lambda x: x[1], reverse=True)

        return [x[0] for x in iou_list]

    def check_head(self, bbox, head):
        if len(self.track) == 0:
            self.track[0] = [(head, bbox)]
            return 0

        for id in self.sorted_iou(bbox):
            dist = self.getDistance(bbox, self.track[id][-1][1])

            if dist > self.max_gap:
                continue

            tst = np.concatenate((head, head), axis=3)
            sim = self.sim_check(tst)

            if sim > self.min_sim:
                self.track[id].append(head, bbox)
                return id

        self.track[len(self.track)] = [(head, bbox)]

        return len(self.track)

    def __call__(self, bboxes, img):
        out = []

        for bbox in bboxes:

            head = self.head_cropping(bbox, img)

            id = self.check_head(bbox, head)

            out.append([*list(bbox[:4]), id])

        return out
