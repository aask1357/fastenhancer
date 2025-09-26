# Dataset
This section describes about dataset preparation.  
For training, we need four types of dataset for:  
- Training
- Validation
- Inferencing (During training, every `infer.interval` epochs, we perform inference for a small amount of data and write the tensorboard log).
- Calculating objective metrics (We re-use valid set).

## Supported Datasets
- [Voicebank-Demand](voicebank-demand.md)
- [DNSChallenge](dns-challenge.md)
