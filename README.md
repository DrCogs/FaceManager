# Face Feature Manager v2.0

基于 **SeetaFace6** 的人脸特征提取与管理工具，提供批量特征提取、人员管理和数据库存储功能。

## 功能特性

- **批量特征提取**：遍历数据集目录，自动从子文件夹（文件夹名 = 人员姓名）中的图片提取人脸特征
- **人员管理**：支持按姓名重命名和删除人员记录
- **数据库概览**：查看当前数据库的人员统计信息
- **持久化存储**：特征数据库以自定义二进制格式保存，支持版本校验与异常恢复

## 项目结构

```
.
├── FaceManager.cpp          # 主程序源码
├── models/                  # SeetaFace6 模型文件目录
│   ├── face_detector.csta           # 人脸检测模型
│   ├── face_landmarker_pts5.csta    # 5点人脸关键点模型
│   └── face_recognizer.csta        # 人脸特征提取模型
└── face_features.db         # 人脸特征数据库（自动生成）
```

## 环境要求

| 依赖项           | 说明                          |
| ---------------- | ----------------------------- |
| C++17 或更高     | 使用了 `std::filesystem` 等特性 |
| OpenCV           | 用于图像读取（`cv::imread`）   |
| SeetaFace6       | 人脸检测、关键点检测、特征提取 |
| CMake            | 构建系统（推荐）              |

### 运行时文件

程序运行前需确保以下文件就位：

- `models/face_detector.csta` — 人脸检测模型
- `models/face_landmarker_pts5.csta` — 人脸关键点检测模型
- `models/face_recognizer.csta` — 人脸特征提取模型

### 支持的图片格式

`.jpg` / `.jpeg` / `.png` / `.bmp` / `.tiff` / `.tif` / `.webp`

## 使用说明

### 启动方式

```
# 使用默认路径
./FaceManager

# 指定模型路径
./FaceManager <检测模型> <关键点模型> <特征提取模型>

# 同时指定模型和数据库路径
./FaceManager <检测模型> <关键点模型> <特征提取模型> <数据库路径>
```

### 数据集目录约定

批量提取时，数据集目录结构如下（文件夹名即为人员姓名）：

```
dataset_root/
├── 张三/
│   ├── photo1.jpg
│   └── photo2.png
├── 李四/
│   └── photo1.jpg
└── ...
```

### 菜单操作

| 选项 | 功能                       |
| ---- | -------------------------- |
| 1    | 批量提取人脸特征           |
| 2    | 按姓名删除人员             |
| 3    | 重命名人员                 |
| 4    | 查看数据库概览             |
| 0    | 退出并保存数据库           |

## 依赖库

本项目依赖以下第三方库：

### SeetaFace6
- 开源人脸识别算法库（人脸检测 / 关键点 / 识别 / 活体 / 口罩检测等）
- 开发包下载地址见官方 README：
  https://github.com/seetafaceengine/SeetaFace6

### OpenCV contrib
- 需安装带 `contrib` 模块的 OpenCV
- 下载地址：https://github.com/opencv/opencv_contrib
