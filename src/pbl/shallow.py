import os
import hydra
from omegaconf import DictConfig
from rich import print as richprint
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.io import loadmat
import sklearn.metrics as metrics
import joblib
from tqdm import tqdm
import imageio
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

os.environ['HYDRA_FULL_ERROR'] = '1'

def load_images(path, test_image):
    images = []
    labels = []
    for filename in os.listdir(path):
        # check if filename contains "_"
        if '_' in filename:
            continue
        elif test_image in filename:
            continue
        else:
            data = loadmat(os.path.join(path, filename))
            # drop noisy bands
            # img = np.delete(data['img'], [103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 221, 222, 223], axis=2)
            img = data['img']
            label = data['map']
            # iterate over the image 
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    images.append(img[i, j])
                    labels.append(label[i, j])
    images = np.array(images)
    labels = np.array(labels)
    if os.path.exists('cache/mean.txt') and os.path.exists('cache/std.txt'):
        # load the mean and std
        mean = np.loadtxt('cache/mean.txt')
        std = np.loadtxt('cache/std.txt')
    else:
        mean = np.mean(images, axis=0)
        std = np.std(images, axis=0)
        np.savetxt('cache/mean.txt', mean)
        np.savetxt('cache/std.txt', std)
    # normalize the images
    images = (images - mean) / std
    return images, labels

def load_test_image(path, test_image):
    data = loadmat(os.path.join(path, test_image))
    img = data['img']
    label = data['map']
    images = []
    labels = []
    # drop noisy bands
    #img = np.delete(img, [103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 221, 222, 223], axis=2)
    shape = img.shape[:2]
    # normalize the image
    mean = np.loadtxt('cache/mean.txt')
    std = np.loadtxt('cache/std.txt')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            images.append(img[i, j])
            labels.append(label[i, j])
    images = np.array(images)
    labels = np.array(labels)
    images = (images - mean) / std
    return images, labels, label

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    richprint(cfg)
    X_train, y_train = load_images(cfg.dataset.path, cfg.dataset.test_image)
    X_test, y_test, full_test_label = load_test_image(cfg.dataset.path, cfg.dataset.test_image)
    # apply pca
    pca = PCA(n_components=32)
    X_train = pca.fit_transform(X_train)
    # train random forest classifier
    # X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    # clf = RandomForestClassifier(n_estimators=600, random_state=42, verbose=True, n_jobs=32)
    # clf = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=True, )
    clf = XGBClassifier(n_estimators=1000, random_state=42, n_jobs=32)
    #clf = IsolationForest(n_estimators=100, random_state=42, verbose=False, n_jobs=16)
    clf.fit(X_train, y_train)
    # save model
    os.makedirs('cache', exist_ok=True)
    joblib.dump(clf, 'cache/model.pkl')
    # load model
    clf = joblib.load('cache/model.pkl')
    # apply pca to test image
    X_test = pca.transform(X_test)
    # predict
    y_pred = clf.predict(X_test)
    # compute iou
    iou = metrics.jaccard_score(y_test, y_pred)
    print(f'IOU: {iou}')
    # compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    # save predictions with imageio
    # multiply by 255 to get the original values
    y_pred = (y_pred * 255)
    # create empty canvas to store the predictions
    test_pred = np.zeros(full_test_label.shape)
    # iterate over the image
    for i in range(test_pred.shape[0]):
        for j in range(test_pred.shape[1]):
            test_pred[i, j] = y_pred[i * test_pred.shape[1] + j]
    # save predictions with imageio
    test_pred = test_pred.astype(np.uint8)
    imageio.imwrite('cache/test_pred.png', test_pred)
    # also save the test_label
    test_label = (full_test_label * 255).astype(np.uint8)
    imageio.imwrite('cache/test_label.png', test_label)
    
    
if __name__ == "__main__":
    main()
