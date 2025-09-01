import torchvision.transforms
from OLD_FILES_DELETE_BEFORE_SUB.jh_data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader
from OLD_FILES_DELETE_BEFORE_SUB.jh_transforms import ToTensor, ScaleAndShiftX, ScaleY

if __name__ == "__main__":
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx",
                                 sequence_length=3,
                                 transform=ToTensor(),
                                 keep_date=False)



    composed = torchvision.transforms.Compose([
        ToTensor(),
        ScaleAndShiftX(feature_index=40, divisor=dataset.max_day),
        ScaleAndShiftX(feature_index=41, divisor=dataset.max_time),
        ScaleY(divisor=dataset.max_tfv, minAfterDiv=0.25, maxAfterDiv=0.75)])

    dataset.set_transform(composed)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=16,
                              shuffle=True)
    """
    first = dataset[0]
    features, labels = first
    print(type(features), type(labels))
    print(features.shape, labels.shape)
    print(features, labels)
    """
    total_samples = len(dataset)

    for i, (input, labels) in enumerate(train_loader):
        print(labels.shape)
        break



    print("Fin")