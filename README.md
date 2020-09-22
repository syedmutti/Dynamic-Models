# Documentation

All the Notebooks have comments within the code. General description of Folders and some of the Notebooks has been mentioned below

## Folder Description

| Folder Names | Description |
| ------ | ------ |
| Trained_Models | Trained Model Weights for Long Bone to Height Prediction |
| Virtual_Segment | Contains Model Scripts(Notenooks) for Height to Long Bones |
| Virtual_Segment/Trained_Models | Trained Models for Height to Long Bones Prediction |
| Extra_Networks | Contains models that are not being used in the Combined Main Model |
| data | Datasets being used |

## NotBooks Description

| NoteBook Names | Description |
| ------ | ------ |
| Combined_model_LongBones_toHeightOnly.ipynb | Model that predicts Height from any Long Bone |
| Combined_model_with_longbones_and_Virtual_Segments.ipynb | Model that predicts Height from any Long Bone & then predicts all the long bones from this height |
| Evaluation_Virtual_Segments.ipynb |  Model that Evaluates Error between measured long bone and predicted long bone after reconstructing it using Height |
| LowerARM_to_HEIGHT.ipynb | Model to predict Height from Lower Arm (Ulna) |
| LowerLEG_to_HEIGHT.ipynb | Model to predict Height from Lower Leg (Tibia) |
| Upper_ARM_to_HEIGHT.ipynb | Model to predict Height from Upper Arm (Humerus) |
| UPPERLEG_HEIGHT.ipynb | Model to predict Height from Upper Leg (Femur) |