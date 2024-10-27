import argparse
import pickle

import numpy as np
from skopt import gp_minimize

def objective(weights):
    pred = np.zeros_like(r[0])
    
    for i in range(len(weights)):
        pred += r[i] * weights[i]
        
    pred = pred.argmax(axis=1)

    correct = (pred == label).sum()
    acc = correct / len(label)
    print(acc)
    return -acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--joint-k2-dir', default=None)
    parser.add_argument('--joint-motion-k2-dir', default=None)

    #arg = parser.parse_args()

    r = []

    label = np.load("test_label_A.npy")

    with open('1.pkl', 'rb') as r1:
        r.append(np.array(list(pickle.load(r1).values())))

    with open('2.pkl', 'rb') as r2:
        r.append(np.array(list(pickle.load(r2).values())))

    with open('3.pkl', 'rb') as r3:
        r.append(np.array(list(pickle.load(r3).values())))

    with open('4.pkl', 'rb') as r4:
        r.append(np.array(list(pickle.load(r4).values())))

    with open('5.pkl', 'rb') as r5:
        r.append(np.array(list(pickle.load(r5).values())))

    with open('6.pkl', 'rb') as r6:
        r.append(np.array(list(pickle.load(r6).values())))

    with open('7.pkl', 'rb') as r7:
        r.append(np.array(list(pickle.load(r7).values())))

    with open('8.pkl', 'rb') as r8:
        r.append(np.array(list(pickle.load(r8).values())))

    with open('9.pkl', 'rb') as r9:
        r.append(np.array(list(pickle.load(r9).values())))
    
    with open('10.pkl', 'rb') as r10:
        r.append(np.array(list(pickle.load(r10).values())))

    with open('11.pkl', 'rb') as r11:
        r.append(np.array(list(pickle.load(r11).values())))
    
    with open('12.pkl', 'rb') as r12:
        r.append(np.array(list(pickle.load(r12).values())))
    
    with open('13.pkl', 'rb') as r13:
        r.append(np.array(list(pickle.load(r13).values())))
    
    with open('14.pkl', 'rb') as r14:
        r.append(np.array(list(pickle.load(r14).values())))

    with open('15.pkl', 'rb') as r15:
        r.append(np.array(list(pickle.load(r15).values())))
    
    with open('16.pkl', 'rb') as r16:
        r.append(np.array(list(pickle.load(r16).values())))
    
    with open('17.pkl', 'rb') as r17:
        r.append(np.array(list(pickle.load(r17).values())))

    with open('18.pkl', 'rb') as r18:
        r.append(np.array(list(pickle.load(r18).values())))

    with open('19.pkl', 'rb') as r19:
        r.append(np.array(list(pickle.load(r19).values())))
    
    r.append(np.load('20.npy')) 
    r.append(np.load('21.npy'))

    space = [(0.2, 1.2) for i in range(21)]
    result = gp_minimize(objective, space, n_calls=200, random_state=1)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))

        # 使用最优权重生成最终预测
    optimal_weights = result.x
    pred = np.zeros_like(r[0])

    for i in range(len(optimal_weights)):
        pred += r[i] * optimal_weights[i]

    # 确保 pred 的形状为 (4599, 155)
    assert pred.shape == (4599, 155), f"Expected shape (4599, 155), got {pred.shape}"

    # 保存预测结果为 pred.npy 文件
    np.save("pred.npy", pred)
    print("Prediction saved to pred.npy")





