from imagenet_x import load_annotations, get_factor_accuracies, FACTORS

def test_load_annotations():
    annotations = load_annotations()
    assert annotations.file_name.is_unique
    assert annotations[FACTORS].notna().all().all()
    
def test_compute_factor_accuracies():
    annotations = get_factor_accuracies('model_predictions/base')
    