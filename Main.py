import os
import pickle
import python_speech_features as psf
import scipy.io.wavfile as wav
import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from tqdm import tqdm

def extract_features(directory, save_file):
    features = {}
    for person in tqdm(os.listdir(directory), desc="Extracting features"):
        person_dir = os.path.join(directory, person)
        if os.path.isdir(person_dir):
            features[person] = {}
            for wav_file in os.listdir(person_dir):
                if wav_file.endswith(".wav"):
                    file_path = os.path.join(person_dir, wav_file)
                    _, signal = wav.read(file_path)
                    mfcc_feat = psf.mfcc(signal, winfunc=np.hamming)
                    features[person][wav_file] = mfcc_feat

    with open(save_file, 'wb') as f:
        pickle.dump(features, f)

extract_features('train_voices', 'template_features.pkl')
extract_features('input_voices', 'input_features.pkl')

def compare_with_same_template(input_features, template_features):
    accuracy = {}
    for person in tqdm(input_features, desc="Comparing with same template"):
        accuracy[person] = {"correct": 0, "total": 0}
        for vowel in input_features[person]:
            mfcc_person_vowel = input_features[person][vowel]
            best_match = None
            best_distance = float('inf')
            correct_vowel = vowel.split('.')[0]

            for other_person in template_features:
                if other_person != person:
                    continue
                else:
                    for other_vowel in template_features[other_person]:
                        mfcc_other_vowel = template_features[other_person][other_vowel]
                        dist, _ = fastdtw.fastdtw(mfcc_person_vowel, mfcc_other_vowel, dist=euclidean)
                        if dist < best_distance:
                            best_distance = dist
                            best_match = other_vowel.split('.')[0]

            if best_match == correct_vowel:
                accuracy[person]["correct"] += 1
            accuracy[person]["total"] += 1

    return accuracy

def compare_with_other_template(input_features, template_features):
    accuracy = {}
    for person in tqdm(input_features, desc="Comparing with other template"):
        accuracy[person] = {"correct": 0, "total": 0}
        for vowel in input_features[person]:
            mfcc_person_vowel = input_features[person][vowel]
            best_match = None
            best_distance = float('inf')
            correct_vowel = vowel.split('.')[0]

            for other_person in template_features:
                if other_person == person:
                    continue
                else:
                    for other_vowel in template_features[other_person]:
                        mfcc_other_vowel = template_features[other_person][other_vowel]
                        dist, _ = fastdtw.fastdtw(mfcc_person_vowel, mfcc_other_vowel, dist=euclidean)
                        if dist < best_distance:
                            best_distance = dist
                            best_match = other_vowel.split('.')[0]

            if best_match == correct_vowel:
                accuracy[person]["correct"] += 1
            accuracy[person]["total"] += 1

    return accuracy

with open('template_features.pkl', 'rb') as f:
    template_features = pickle.load(f)

with open('input_features.pkl', 'rb') as f:
    input_features = pickle.load(f)

self_accuracy = compare_with_same_template(input_features, template_features)
print('\n')

for person in self_accuracy:
    correct = self_accuracy[person]["correct"]
    total = self_accuracy[person]["total"]
    acc = correct / total
    print(f"Accuracy for {person}: {acc:.2%}")

print("Average accuracy: {:.2%}".format(sum([self_accuracy[person]["correct"] / self_accuracy[person]["total"] for person in self_accuracy]) / len(self_accuracy)))

other_accuracy = compare_with_other_template(input_features, template_features)
print('\n')

for person in other_accuracy:
    correct = other_accuracy[person]["correct"]
    total = other_accuracy[person]["total"]
    acc = correct / total
    print(f"Accuracy for {person}: {acc:.2%}")

print("Average accuracy: {:.2%}".format(sum([other_accuracy[person]["correct"] / other_accuracy[person]["total"] for person in other_accuracy]) / len(other_accuracy)))