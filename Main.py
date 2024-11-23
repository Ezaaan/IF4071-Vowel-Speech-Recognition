import os
import pickle
import python_speech_features as psf
import scipy.io.wavfile as wav
import fastdtw
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter
from dtaidistance import dtw, dtw_ndim
import numpy as np
from tqdm import tqdm

def pre_emphasis(signal, alpha=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal

def averaging_template(template):
    avg_template = [np.mean(segment, axis=0) for segment in segmental_k_means(vectors=template, K=3, max_iterations=10)]
    return avg_template

def uniform_segmentation(data, k):
    n = len(data)
    segment_size = n // k 
    remainder = n % k

    segments = []
    points = []
    start = 0
    end = -1
    
    for i in range(k):
        if end >= 0:
            points.append(end - 1)
        end = start + segment_size + (1 if remainder > 0 else 0)
        segments.append(data[start:end])
        start = end
        remainder -= 1

    return segments, points

def segmental_k_means(vectors, K, max_iterations=10):
    segmented_vectors, initial_cluster_points = uniform_segmentation(vectors, K)
    cluster_points = [np.mean(segment, axis=0) for segment in segmented_vectors]
    new_segments = [[] for _ in range(K)]

    i = 0
    while i < max_iterations:
        current_point = 0
        current_segment = 0
        max_seg = current_segment + 1 if current_segment + 1 < K else K-1
        seg_nums = [i for i in range(current_segment, max_seg+1)]
        new_cluster_points = []
        new_segments = [[] for _ in range(K)]

        j = 0
        while j < len(vectors):
            if current_point < len(initial_cluster_points) and j == initial_cluster_points[current_point]:
                no_change = True
                while no_change:
                    distances = [np.linalg.norm(vectors[j] - cluster_points[model]) for model in range(current_segment, max_seg+1)]
                    closest_segment = seg_nums[np.argmin(distances)]
                    new_segments[closest_segment].append(vectors[j])

                    if closest_segment == current_segment:
                        j += 1

                    else:
                        no_change = False
                        new_cluster_points.append(j-1)
                        current_segment += 1
                        current_point += 1
                        max_seg = current_segment + 1 if current_segment + 1 < K else K-1
                        seg_nums = [i for i in range(current_segment, max_seg+1)]
            else:
                new_segments[current_segment].append(vectors[j])
            j += 1

        initial_cluster_points = new_cluster_points
        cluster_points = [np.mean(segment, axis=0) for segment in new_segments]
        
        if all(np.array_equal(new_segments[i], segmented_vectors[i]) for i in range(K)):
            break
        segmented_vectors = [seg for seg in new_segments]

        i += 1
        
    segmented_vectors = [seg for seg in new_segments]
    
    return segmented_vectors

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
                    signal = pre_emphasis(signal)
                    mfcc_feat = psf.mfcc(signal, numcep=13, winfunc=np.hamming)
                    result = gaussian_filter(mfcc_feat, sigma=1)

                    # feat_mfcc_d = psf.delta(mfcc_feat, 2)
                    # feat_mfcc_dd = psf.delta(feat_mfcc_d, 2)
                    # result = np.column_stack((mfcc_feat, feat_mfcc_d, feat_mfcc_dd))

                    result = averaging_template(np.array(result))

                    features[person][wav_file] = np.array(result)

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
                        # dist, _ = fastdtw.fastdtw(mfcc_person_vowel, mfcc_other_vowel, dist=euclidean)
                        print(type(mfcc_person_vowel), type(mfcc_other_vowel))
                        dist = dtw_ndim.distance_fast(mfcc_person_vowel, mfcc_other_vowel)
                        if dist < best_distance:
                            best_distance = dist
                            best_match = other_vowel.split('.')[0]

            # print(f"Predicted: {person}-{best_match}, Actual: {person}-{correct_vowel}")
            if best_match == correct_vowel:
                accuracy[person]["correct"] += 1
            accuracy[person]["total"] += 1

    return accuracy

def compare_with_other_template(input_features, template_features):
    accuracy = {}
    for person in tqdm(input_features, desc="Comparing with other template"):
        accuracy[person] = {}
        
        for other_person in template_features:
            if other_person != person:
                accuracy[person][other_person] = {"correct": 0, "total": 0, "accuracy": 0}

        for vowel in input_features[person]:
            mfcc_person_vowel = input_features[person][vowel]
            correct_vowel = vowel.split('.')[0]

            for other_person in template_features:
                if other_person == person:
                    continue 
                best_match = None
                best_distance = float('inf')

                for other_vowel in template_features[other_person]:
                    mfcc_other_vowel = template_features[other_person][other_vowel]
                    # dist, _ = fastdtw.fastdtw(mfcc_person_vowel, mfcc_other_vowel, dist=euclidean)
                    dist = dtw_ndim.distance_fast(mfcc_person_vowel, mfcc_other_vowel)

                    if dist < best_distance:
                        best_distance = dist
                        best_match = other_vowel.split('.')[0]

                # print(f"Predicted: {other_person}-{best_match}, Actual: {person}-{correct_vowel}")
                if best_match == correct_vowel:
                    accuracy[person][other_person]["correct"] += 1
                accuracy[person][other_person]["total"] += 1

                accuracy[person][other_person]["accuracy"] = (
                    accuracy[person][other_person]["correct"] / accuracy[person][other_person]["total"]
                )

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
    avg_acc = sum([other_accuracy[person][other_person]["accuracy"] for other_person in other_accuracy[person]]) / len(other_accuracy[person])
    print(f"Average accuracy for {person}: {avg_acc:.2%}")

    for other_person in other_accuracy[person]:
        print(f"Accuracy for {person} using {other_person}'s templates: {other_accuracy[person][other_person]['accuracy']:.2%}")

total_accuracy_sum = sum(
    [sum([other_accuracy[person][other_person]["accuracy"] for other_person in other_accuracy[person]]) for person in other_accuracy]
)
total_comparisons = sum([len(other_accuracy[person]) for person in other_accuracy])
overall_avg_accuracy = total_accuracy_sum / total_comparisons

print("Overall average accuracy: {:.2%}".format(overall_avg_accuracy))
