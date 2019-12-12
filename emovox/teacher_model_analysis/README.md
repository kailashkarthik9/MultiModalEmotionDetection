# EmoVoxCelebDatasetAnalysis
Analysis of the EmoVoxCeleb dataset labels

### Observations

* As observed in the EmoVoxCeleb website, there are many utterances where apart from the prominent emotion, the neutral emotion also scores high as the image classification algorithm detects the emotions in the transition frames as neutral

* The difference between the highest and second highest scoring emotions across an utterance has an unstable statistical distribution. There are a lot of utterances where the two emotions score very closely and it might be difficult to choose one decisively unless one of the emotions is neutral.

* The distributions across a single emotion is also not very stable and the standard deviations are scattered.

* A final analysis was done by choosing just the highest scoring emotion and the same as above was observed.

### Files

##### Directory - emotion_statistics
* **emotion_** files are the distribution of standard deviations across utterances for each emotion

##### Directory - best_emotion_statistics

* **best_emotion_using_means** is the distribution of standard deviations across utterances for the emotion with the highest mean value for that utterance
* **best_emotion_using_max** is the distribution of standard deviations across utterances for the emotion with the highest max value for that utterance

##### Directory - utterance_statistics

* **means** is the distribution of standard deviations of means across emotions for each utterance
* **maxs** is the distribution of standard deviations of maximum values across emotions for each utterance
* **max_diffs** is the distribution of standard deviations of difference between the two top scoring emotions across utterances
