from numpy import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plot
from sklearn.decomposition import FastICA
import cv2

# 画像を保存しているフォルダのパス
image_path = 'C:/Users/Owner/Pictures/'

# ソース画像読み込み
src_lenna_image = cv2.imread(image_path + 'lenna.png')
src_fruits_image = cv2.imread(image_path + 'fruits.png')

# 画像サイズ、チャンネルが一致しているかチェック
tmp = [src_lenna_image, src_fruits_image]
if tmp[0].shape != tmp[1].shape:
    print("入力されている画像サイズが一致しませんので整形します")
    size = (512, 512)
    for i in range(len(np.asarray(tmp))):
        tmp[i] = cv2.resize(tmp[i], size)

    src_lenna_image = tmp[0]
    src_fruits_image = tmp[1]
    print(" -> 画像サイズが512*512に調整されました")

# ソース画像をグレースケールに変換
gray_lenna = cv2.cvtColor(src_lenna_image, cv2.COLOR_RGB2GRAY)
gray_fruits = cv2.cvtColor(src_fruits_image, cv2.COLOR_RGB2GRAY)
# mix_image = cv2.addWeighted(gray_lenna, 0.7, gray_fruits, 0.3, 0)

# 2次元配列を1次元ベクトルに変換する(512*512) -> (262144, )
gray_lenna_flat = np.asarray(gray_lenna).flatten()
gray_fruits_flat = np.array(gray_fruits).flatten()

# 2つのベクトルを結合(262144, 2)
S = np.c_[gray_lenna_flat, gray_fruits_flat]  # ソース信号

# (lenna)結合したベクトルを分離して2次元配列に再構成し、元配列と一致するかテスト
print('Gray Lenna Vector(Flatten) = {}'.format(S[::1, 0:1].T))  # Sの第１列ベクトルのみを表示
S1_tmp = np.reshape(S[::1, 0:1].T, (512, 512))  # Sを512*512の二次元配列に変換
cv2.imshow('S1_tmp', S1_tmp)
assert np.allclose(gray_lenna, S1_tmp)  # 元の白黒レナ画像と再構成した2次元配列は等しい

# (fruits)結合したベクトルを分離して2次元配列に再構成し、元配列と一致するかテスト
print('Gray Fruits Vector(Flatten) = {}'.format(S[::1, 1:2].T))  # Sの第２列ベクトルのみを表示
S2_tmp = np.reshape(S[::1, 1:2].T, (512, 512))  # Sを512*512の二次元配列に変換
cv2.imshow('S2_tmp', S2_tmp)
assert np.allclose(gray_fruits, S2_tmp)  # 元の白黒野菜画像と再構成した2次元配列は等しい

# 人工的にソース信号にノイズを加え混合する
S = S.astype(np.float64)
# noise = 0.2 * np.random.normal(size=S.shape)
# S += noise  # ノイズを加える
S /= S.std(axis=0)  # データの正規化

A = np.array([[1.2, 0.4], [0.8, 1.5]])  # 混合行列
A = np.array([[1.0, 0.0], [0.0, 1.0]])  # 混合行列
X = np.dot(S, A.T)

cv2.imwrite(image_path + 'mix_image.png', X)

# 混合行列に対して独立成分分析を適用する
decomposer = FastICA()
decomposer.fit(X)
S_ = decomposer.transform(X)  # 信号の再構成
A_ = decomposer.mixing_  # 混合行列の推定

X = np.dot(S_, A_.T) + decomposer.mean_
X = np.round(X).astype(uint8)  # float64 -> uint8に変換

# 観測信号(配列)を画像化する
mix_1 = (np.reshape(X[::1, 0:1].T, (512, 512))) * 15
mix_2 = (np.reshape(X[::1, 1:2].T, (512, 512))) * 15

mix_images = [mix_1, mix_2]

mix_image_names = ['mix_1', 'mix_2']
for i in range(len(np.asarray(mix_images))):
    cv2.imshow(mix_image_names[i], mix_images[i])

print('X(Observation signal) Property')
print(' shape:{}, type = {}'.format(X.shape, type(X)))
print(' Component Type = {}'.format(X[1]), end='\n\n')

# print(S_)
# print(np.min(S_))
# print(abs(np.min(S_)))
# print(S_ + abs(np.min(S_)))
# print(np.max(S_ + abs(np.min(S_))))

min = np.min(S_)
max = np.max(S_)
result = (S_ - min) / (max - min)

figure(1)
plot.plot(S_, label="S_")
plot.legend()

figure(2)
plot.plot(result, label="result")
plot.legend()

# print("max : {}, min : {}".format(np.max(result), np.min(result)))
# print(result)
# print("result", result * 255)
# print("gray lenna", gray_lenna_flat.T)

result *= 255
S_ica_fruits = np.round(result[::1, 0:1]).astype(np.uint8)
S_ica_lenna = np.round(result[::1, 1:2]).astype(np.uint8)
# S_ica_lenna = np.round(S_[::1, 0:1].T * pow(10, 5)).astype(np.uint8)
# S_ica_fruits = np.round(S_[::1, 1:2].T * pow(10, 5)).astype(np.uint8)

cv2.imshow('S_ica_lenna', np.reshape(S_ica_fruits, (512, 512)))
cv2.imshow('S_ica_fruits', np.reshape(S_ica_lenna, (512, 512)))
# cv2.imshow('S_ica_fruits', np.reshape(S_ica_fruits, (512, 512)))

# print(S_ica_lenna)
# print(S_ica_fruits)
# S_ica_recovered = S_


models = [X, S, S_]
names = ['Observations', 'True Sources', 'ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

figure(3)
for ii, (model, name) in enumerate(zip(models, names), 1):
    plot.subplot(3, 1, ii)
    plot.title(name)
    for sig, color in zip(model.T, colors):
        plot.plot(sig, color=color)

plot.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plot.show()

# cv2.imshow('dst', X)
# cv2.imshow('mix_image', X)

# # スパースコーディング
# decomposer = DictionaryLearning()
# decomposer.fit(mix_array.T)
# Usc = decomposer.components_.T
# Asc = decomposer.transform(mix_array.T).T

# axis('equal')
# plot(mix_array[0], mix_array[1], 'xc')
# # Upca = Upca / sqrt((Upca ** 2).sum(axis=0))
# Uica = Uica / sqrt((Uica ** 2).sum(axis=0))
# # Usc = Usc / sqrt((Usc ** 2).sum(axis=0))
# for i in range(2):
#     # p_pca = plot([0, Upca[0, i]], [0, Upca[1, i]], '-r')
#     p_ica = plot([0, Uica[0, i]], [0, Uica[1, i]], '-b')
#     # p_sc = plot([0, Usc[0, i]], [0, Usc[1, i]], '-g')
# legend(('data', 'PCA', 'ICA', 'SC'))
# legend(loc="best", prop=dict(size=12))
# show()
#
# subplot(1, 3, 1)
# # plot(Apca[0], Apca[1], 'xc')
# title('PCA')
# subplot(1, 3, 2)
# plot(Aica[0], Aica[1], 'xc')
# title('ICA')
# subplot(1, 3, 3)
# # plot(Asc[0], Asc[1], 'xc')
# title('SC')
# show()

# 画像の描画
# cv2.imshow('lenna', src_lenna_image)
# cv2.imshow('fruits', src_fruits_image)
# cv2.imshow('dst', mix_image)

# キー入力待ち、ウィンドウ破棄
cv2.waitKey(0)
cv2.destroyAllWindows()
