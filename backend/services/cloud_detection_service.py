from ai.cloud_detector.evaluate import predict_mask


def detect_clouds(image_path: str):

    result = predict_mask(
        image_path=image_path
    )

    return result