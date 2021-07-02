import cv2
import numpy as np


class PoseEstimationModel():
    def __init__(self) -> None:
        self.net = cv2.dnn.readNetFromTensorflow("./models/graph_opt.pb")
        self.min_conf = 0.2

        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                           "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                           "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                           ["RElbow", "RWrist"], ["LShoulder",
                                                  "LElbow"], ["LElbow", "LWrist"],
                           ["Neck", "RHip"], ["RHip", "RKnee"], [
            "RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], [
            "Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        self.inWidth = 368
        self.inHeight = 368

    def infer(self, img):
        frameHeight, frameWidth, channels = img.shape
        self.net.setInput(cv2.dnn.blobFromImage(
            img, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.net.forward()
        # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
        out = out[:, :19, :, :]

        assert(len(self.BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confiden)ce is higher than threshold.
            points.append((int(x), int(y)) if conf > self.min_conf else None)

        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in self.BODY_PARTS)
            assert(partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(img, points[idFrom], (3, 3),
                            0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(img, points[idTo], (3, 3),
                            0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = self.net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(img, '%.2fms' % (t / freq), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        return img, ['person']
