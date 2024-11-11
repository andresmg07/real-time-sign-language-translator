# Main execution module.
import numpy as np
from feature_extractor import video_feature_extraction
from validator import get_support_vector_machine_model, load_object

if __name__ == '__main__':
    print("ASL recognizer.")
    vectors = load_object("./features")
    classifier = get_support_vector_machine_model(vectors)
    video_feature_extraction(classifier)
