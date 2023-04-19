# Sound Event Detection with Soft Labels

This task is a subtopic of the Sound event detection task (task 4) which provides for training weakly labeled data (without timestamps), strongly-labeled synthetic data (with timestamps) and unlabeled data. The target of the systems is to provide not only the event class but also the event time localization given that multiple events can be present in an audio recording (see also Fig 1).

Specific to this subtask is another type of training data:

- Soft labels provided as a number between 0 and 1 that characterize the certainty of human annotators for the sound at that specific time

- The temporal resolution of the provided data is 1 second (due to the annotation procedure)

- Systems will be evaluated against hard labels, obtained by thresholding the soft labels at 0.5; anything above 0.5 is considered 1 (sound active), anything below 0.5 is considered 0 (sound inactive)

Research question: Do soft labels contain any useful additional information to help train better sound event detection systems?

https://dcase.community/challenge2023/task-sound-event-detection-with-soft-labels

### DATASET

MAESTRO Real - Multi-Annotator Estimated Strong Labels: https://zenodo.org/record/7244360

![Alt text](metadata/gt.png?raw=true "Title")

### Usage

Extract Feature: ...

```python extract_feature.py```

Train model:

```python train.py```

Test model with F1-Macro (Threshold = 0.5 for all):

```python validation.py```

Test model with F1-Macro-Optimize (threshold for each class):

```python test.py```

Data visualization:

```python visualization.py```