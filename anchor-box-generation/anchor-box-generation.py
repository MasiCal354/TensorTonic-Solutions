def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    anchors = []

    for i in range(feature_size):
        cy = (i + 0.5) * stride
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)

                    anchors.append([
                        cx - w / 2.0,
                        cy - h / 2.0,
                        cx + w / 2.0,
                        cy + h / 2.0
                    ])
    return anchors