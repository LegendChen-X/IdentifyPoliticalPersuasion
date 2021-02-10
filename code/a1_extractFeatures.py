#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import string
import re

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
    
CDF = 0

if not CDF:
    BGLpath = "./../Wordlists/BristolNorms+GilhoolyLogie.csv"
    WARpath = "./../Wordlists/Ratings_Warriner_et_al.csv"
    altPathData = "./../feats/Alt_IDs.txt"
    leftPathData = "./../feats/Left_IDs.txt"
    rightPathData = "./../feats/Right_IDs.txt"
    centerPathData = "./../feats/Center_IDs.txt"
    altPathArray = "./../feats/Alt_feats.dat.npy"
    leftPathArray = "./../feats/Left_feats.dat.npy"
    rightPathArray = "./../feats/Right_feats.dat.npy"
    centerPathArray = "./../feats/Center_feats.dat.npy"
else:
    BGLpath = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
    WARpath = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
    altPathData = "/u/cs401/A1/feats/Alt_IDs.txt"
    leftPathData = "/u/cs401/A1/feats/Left_IDs.txt"
    rightPathData = "/u/cs401/A1/feats/Right_IDs.txt"
    centerPathData = "/u/cs401/A1/feats/Center_IDs.txt"
    altPathArray = "/u/cs401/A1/feats/Alt_feats.dat.npy"
    leftPathArray = "/u/cs401/A1/feats/Left_feats.dat.npy"
    rightPathArray = "/u/cs401/A1/feats/Right_feats.dat.npy"
    centerPathArray = "/u/cs401/A1/feats/Center_feats.dat.npy"
    
altArray = np.load(altPathArray)
leftArray = np.load(leftPathArray)
rightArray = np.load(rightPathArray)
centerArray = np.load(centerPathArray)
altIDs = {}
leftIDs = {}
rightIDs = {}
centerIDs= {}
altFp = open(altPathData, "r")
leftFp = open(leftPathData, "r")
rightFp = open(rightPathData, "r")
centerFp = open(centerPathData, "r")
altData = altFp.read().split()
altFp.close()
leftData = leftFp.read().split()
leftFp.close()
rightData = rightFp.read().split()
rightFp.close()
centerData = centerFp.read().split()
centerFp.close()

for i in range(len(altData)): altIDs[altData[i]] = i
for i in range(len(leftData)): leftIDs[leftData[i]] = i
for i in range(len(rightData)): rightIDs[rightData[i]] = i
for i in range(len(centerData)): centerIDs[centerData[i]] = i

fp = open(BGLpath,"r")
BGLdata = fp.read().split("\n")[1:-3]
fp.close()
BGL = {}
for line in BGLdata:
    element = line.split(",")
    BGL[element[1]] = [element[3], element[4], element[5]]
    
fp = open(WARpath, "r")

WARdata = fp.read().split('\n')[1:-1]
WAR = {}
for line in WARdata:
    element = line.split(",")
    WAR[element[1]] = [element[2], element[5], element[8]]

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: Extract features that rely on capitalization.
    features = np.zeros(173)
    data = re.compile("(\S+)/(?=\S+)").findall(comment)
    tags = re.compile("(?<=\S)/(\S+)").findall(comment)
    # TODO: Extract features that rely on capitalization.
    for text in data:
        if text.isupper() and len(text) >= 3: features[0] += 1
        
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    main = [text.lower() for text in data]
    # TODO: Extract features that do not rely on capitalization.
        
    features[4] = tags.count('CC')
    features[5] = tags.count('VBD')
    
    features[6] += len(re.compile(r'\b(' + r'|'.join(['\'ll', 'will', 'shall', 'gonna']) + r')\b').findall(comment))
    
    features[6] += len(re.compile(r"go/VBG to/TO [\w]+/VB").findall(comment))
    
    features[7] = re.compile("(?=/)[\S]+").sub('', comment).count(',')
    
    features[8] = len(re.findall(' \W{2,}/', comment))
    
    features[9] = tags.count('NN') + tags.count('NNS')
    
    features[10] = tags.count('NNP') + tags.count('NNPS')
    
    features[11] = tags.count('RB') + tags.count('RBR') + tags.count('RBS')
    
    features[12] = tags.count('WDT') + tags.count('WP$') + tags.count('WP') + tags.count('WRB')
    
    for text in main:
        if text in FIRST_PERSON_PRONOUNS: features[1] += 1
        if text in SECOND_PERSON_PRONOUNS: features[2] += 1
        if text in THIRD_PERSON_PRONOUNS: features[3] += 1
        if text in SLANG: features[13] += 1
        
    breaks = comment.count('\n')
    
    if not breaks: breaks += 1
    
    features[14] = len(main) / breaks
    
    count = 0
    
    length = 0
    
    for text in main:
        if not set(text).issubset(set(string.punctuation)):
            count += 1
            length += len(text)
            
    if not count: count += 1
    
    if main: features[15] = length / count
    else: features[15] = 0
    
    features[16] = breaks
    
    AoA = []
    IMG = []
    FAM = []
    Vs = []
    As = []
    Ds = []
    
    for text in main:
        if text in BGL:
            AoA.append(BGL[text][0])
            IMG.append(BGL[text][1])
            FAM.append(BGL[text][2])
        if text in WAR:
            Vs.append(WAR[text][0])
            As.append(WAR[text][1])
            Ds.append(WAR[text][2])
            
    AoA = np.asarray(AoA, np.float32)
    IMG = np.asarray(IMG, np.float32)
    FAM = np.asarray(FAM, np.float32)
    Vs = np.asarray(Vs, np.float32)
    As = np.asarray(As, np.float32)
    Ds = np.asarray(Ds, np.float32)
    features[17] = np.mean(AoA)
    features[18] = np.mean(IMG)
    features[19] = np.mean(FAM)
    features[20] = np.std(AoA)
    features[21] = np.std(IMG)
    features[22] = np.std(FAM)
    features[23] = np.mean(Vs)
    features[24] = np.mean(As)
    features[25] = np.mean(Ds)
    features[26] = np.std(Vs)
    features[27] = np.std(As)
    features[28] = np.std(Ds)
    
    return features
    
    
def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    if comment_class == "Alt": feat = np.append(feat[:29], altArray[altIDs[comment_id]])
        
    elif comment_class == "Left": feat = np.append(feat[:29], leftArray[leftIDs[comment_id]])
        
    elif comment_class == "Right": feat = np.append(feat[:29], rightArray[rightIDs[comment_id]])
    
    elif comment_class == "Center": feat = np.append(feat[:29], centerArray[centerIDs[comment_id]])
        
    return feat


def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    
    for i in range(len(data)):
        res = extract1(data[i]["body"])
        if data[i]["cat"] == "Left": feats[i][-1] = 0
        elif data[i]["cat"] == "Center": feats[i][-1] = 1
        elif data[i]["cat"] == "Right": feats[i][-1] = 2
        elif data[i]["cat"] == "Alt": feats[i][-1] = 3
        res = extract2(res, data[i]["cat"], data[i]['id'])
        feats[i,:-1] = res
        
    feats = np.nan_to_num(feats)
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

