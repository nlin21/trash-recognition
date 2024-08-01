# For use with running models hosted on Roboflow
# Make sure to have your API key in the working environment
#   export ROBOFLOW_API_KEY=<your api key>

from inference import get_model
import supervision as sv
import cv2

if __name__ == '__main__':
    im = "samples/sample1.png"
    image = cv2.imread(im)

    model = get_model(model_id="taco-trash-annotations-in-context/16")
    model.confidence = 80
    results = model.infer(image, confidence=0.6)[0]

    detections = sv.Detections.from_inference(results)
    bounding_box_annotator = sv.BoundingBoxAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)

    sv.plot_image(annotated_image)
