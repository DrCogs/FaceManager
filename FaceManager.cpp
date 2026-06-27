/**
 * Face Feature Management System - SeetaFace6 + OpenCV
 * Features: batch extract face features, delete person, view database
 * Build: MSVC + OpenCV + SeetaFace6
 *
 * Usage:
 *   face_manager.exe [detector_model] [landmarker_model] [recognizer_model] [db_file]
 *   Default model path: ./models/
 *   Default db: ./face_features.db
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <filesystem>
#include <opencv2/opencv.hpp>

 // SeetaFace6 headers
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>

namespace fs = std::filesystem;

// =========================== Global Config ===========================
std::string g_detectorModel = "models/face_detector.csta";
std::string g_landmarkerModel = "models/face_landmarker_pts5.csta";
std::string g_recognizerModel = "models/face_recognizer.csta";
std::string g_dbPath = "face_features.db";

// =========================== Data Structures ===========================
struct PersonEntry {
    std::string name;
    std::vector<float> features;
};

// Global feature database
std::vector<PersonEntry> g_database;
int g_featureSize = 0;

// Database file magic number
constexpr uint32_t DB_MAGIC = 0x53464644; // "SFFD"
constexpr uint32_t DB_VERSION = 1;

// =========================== Database I/O ===========================
bool saveDatabase(const std::string& path) {
    if (g_database.size() > 0xFFFFFFFFULL) {
        std::cerr << "Error: record count exceeds storage limit" << std::endl;
        return false;
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot write to file " << path << std::endl;
        return false;
    }

    ofs.write(reinterpret_cast<const char*>(&DB_MAGIC), sizeof(DB_MAGIC));
    ofs.write(reinterpret_cast<const char*>(&DB_VERSION), sizeof(DB_VERSION));
    ofs.write(reinterpret_cast<const char*>(&g_featureSize), sizeof(g_featureSize));

    uint32_t count = static_cast<uint32_t>(g_database.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& entry : g_database) {
        uint32_t nameLen = static_cast<uint32_t>(entry.name.size());
        ofs.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        ofs.write(entry.name.data(), nameLen);
        ofs.write(reinterpret_cast<const char*>(entry.features.data()),
            static_cast<std::streamsize>(g_featureSize * sizeof(float)));
    }

    ofs.close();
    std::cout << "Database saved to " << path << " (" << count << " records)" << std::endl;
    return true;
}

bool loadDatabase(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint32_t magic = 0, version = 0, count = 0;
    int fileFeatureSize = 0;

    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != DB_MAGIC) {
        std::cerr << "Warning: invalid database format" << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != DB_VERSION) {
        std::cerr << "Warning: incompatible database version" << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char*>(&fileFeatureSize), sizeof(fileFeatureSize));
    if (fileFeatureSize <= 0 || fileFeatureSize > 100000) {
        std::cerr << "Warning: abnormal feature dimension (" << fileFeatureSize << ")" << std::endl;
        return false;
    }

    // Set global feature size from file if not yet initialized
    if (g_featureSize == 0) {
        g_featureSize = fileFeatureSize;
    }
    else if (g_featureSize != fileFeatureSize) {
        std::cerr << "Error: DB feature dimension (" << fileFeatureSize
            << ") does not match model (" << g_featureSize
            << "). Please delete old DB and recreate." << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count > 1000000) { // safety: max 1M records
        std::cerr << "Error: abnormal record count (" << count << "), file may be corrupted" << std::endl;
        return false;
    }
    g_database.clear();
    g_database.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nameLen = 0;
        ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        if (nameLen > 1024) { // safety check
            std::cerr << "Error: DB corrupted (name length abnormal)" << std::endl;
            g_database.clear();
            return false;
        }

        PersonEntry entry;
        entry.name.resize(nameLen);
        ifs.read(entry.name.data(), nameLen);

        entry.features.resize(g_featureSize);
        ifs.read(reinterpret_cast<char*>(entry.features.data()),
            static_cast<std::streamsize>(g_featureSize * sizeof(float)));

        if (ifs.fail()) {
            std::cerr << "Error: DB read failed, file may be corrupted" << std::endl;
            g_database.clear();
            return false;
        }

        g_database.push_back(std::move(entry));
    }

    std::cout << "Database loaded: " << path << " (" << g_database.size() << " records)" << std::endl;
    return true;
}

// =========================== Core Functions ===========================
// Extract face features from a single image
// Returns number of features extracted
int extractFromImage(const cv::Mat& imgBGR,
    seeta::FaceDetector& detector,
    seeta::FaceLandmarker& landmarker,
    seeta::FaceRecognizer& recognizer,
    std::vector<std::vector<float>>& outFeatures) {
    SeetaImageData seetaImg;
    seetaImg.data = imgBGR.data;
    seetaImg.width = imgBGR.cols;
    seetaImg.height = imgBGR.rows;
    seetaImg.channels = imgBGR.channels();

    // Face detection
    SeetaFaceInfoArray faces = detector.detect(seetaImg);
    if (faces.size == 0) return 0;

    outFeatures.clear();
    outFeatures.reserve(static_cast<size_t>(faces.size));

    std::vector<float> feature(g_featureSize);

    for (int i = 0; i < faces.size; ++i) {
        const auto& face = faces.data[i];
        SeetaRect faceRect = face.pos;

        // Landmark detection (5 points)
        constexpr int LANDMARK_COUNT = 5;
        SeetaPointF points[LANDMARK_COUNT];
        landmarker.mark(seetaImg, faceRect, points);

        // Feature extraction
        recognizer.Extract(seetaImg, points, feature.data());

        outFeatures.push_back(feature);
    }

    return static_cast<int>(outFeatures.size());
}

// Check if a file is a supported image format
static bool isImageFile(const std::string& ext) {
    std::string lower = ext;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return lower == ".jpg" || lower == ".jpeg" || lower == ".png" ||
        lower == ".bmp" || lower == ".tiff" || lower == ".tif" ||
        lower == ".webp";
}

// Extract features from all images in a single person's folder
// Returns number of records added
static int extractFromFolder(const std::string& folderPath,
    const std::string& name,
    seeta::FaceDetector& detector,
    seeta::FaceLandmarker& landmarker,
    seeta::FaceRecognizer& recognizer) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && isImageFile(entry.path().extension().string())) {
            imagePaths.push_back(entry.path().string());
        }
    }

    if (imagePaths.empty()) {
        std::cout << "  [Person: " << name << "] No images found, skipping" << std::endl;
        return 0;
    }

    std::sort(imagePaths.begin(), imagePaths.end());

    bool isNew = std::none_of(g_database.begin(), g_database.end(),
        [&name](const PersonEntry& e) { return e.name == name; });
    std::cout << "  [Person: " << name << "] " << imagePaths.size() << " images"
        << (isNew ? " (new)" : " (append)") << std::endl;

    int added = 0;
    for (const auto& imgPath : imagePaths) {
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "    Warning: cannot read " << imgPath << ", skipping" << std::endl;
            continue;
        }

        try {
            std::vector<std::vector<float>> features;
            int nFaces = extractFromImage(img, detector, landmarker, recognizer, features);

            if (nFaces == 0) {
                std::cout << "    [no face] " << fs::path(imgPath).filename().string() << std::endl;
                continue;
            }

            std::cout << "    [" << nFaces << " faces] " << fs::path(imgPath).filename().string() << std::endl;

            for (auto& feat : features) {
                PersonEntry entry;
                entry.name = name;
                entry.features = std::move(feat);
                g_database.push_back(std::move(entry));
                ++added;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "    Warning: error processing " << imgPath << ": " << e.what() << ", skipping" << std::endl;
        }
        catch (...) {
            std::cerr << "    Warning: unknown error processing " << imgPath << ", skipping" << std::endl;
        }
    }

    std::cout << "  [Person: " << name << "] Done, added " << added << " record(s)" << std::endl;
    return added;
}

// Batch extract: traverse all subdirs under rootDir, folder name = person name
// Returns true if any data was added
bool batchExtract(const std::string& rootDir,
    seeta::FaceDetector& detector,
    seeta::FaceLandmarker& landmarker,
    seeta::FaceRecognizer& recognizer) {
    if (!fs::exists(rootDir) || !fs::is_directory(rootDir)) {
        std::cerr << "Error: directory not found or invalid: " << rootDir << std::endl;
        return false;
    }

    // Collect person folders (subdirectories)
    std::vector<std::string> personFolders;
    for (const auto& entry : fs::directory_iterator(rootDir)) {
        if (entry.is_directory()) {
            personFolders.push_back(entry.path().string());
        }
    }

    if (personFolders.empty()) {
        std::cerr << "Error: no subdirectories found in " << rootDir << std::endl;
        return false;
    }

    std::sort(personFolders.begin(), personFolders.end());

    std::cout << "\nFound " << personFolders.size() << " person folder(s) in " << rootDir << std::endl;
    std::cout << "==========================================" << std::endl;

    int totalAdded = 0;
    int totalPersons = 0;

    for (const auto& folderPath : personFolders) {
        std::string name = fs::path(folderPath).filename().string();
        int added = extractFromFolder(folderPath, name, detector, landmarker, recognizer);
        if (added > 0) {
            ++totalPersons;
            totalAdded += added;
        }
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Batch extraction done: " << totalPersons << " person(s) updated, "
        << totalAdded << " total record(s) added" << std::endl;
    return totalAdded > 0;
}

// Delete all records for a specific person
int deletePerson(const std::string& name) {
    auto it = std::remove_if(g_database.begin(), g_database.end(),
        [&name](const PersonEntry& e) {
            return e.name == name;
        });
    int removed = static_cast<int>(std::distance(it, g_database.end()));
    g_database.erase(it, g_database.end());
    return removed;
}

// View database contents
void viewDatabase() {
    if (g_database.empty()) {
        std::cout << "Database is empty" << std::endl;
        return;
    }

    // Count features per person
    std::vector<std::pair<std::string, int>> stats;
    for (const auto& entry : g_database) {
        auto it = std::find_if(stats.begin(), stats.end(),
            [&](const auto& p) { return p.first == entry.name; });
        if (it != stats.end()) {
            ++it->second;
        }
        else {
            stats.emplace_back(entry.name, 1);
        }
    }

    std::cout << "\n========== Database Overview ==========" << std::endl;
    std::cout << "Total records: " << g_database.size() << std::endl;
    std::cout << "Persons:       " << stats.size() << std::endl;
    std::cout << "Feature dim:   " << g_featureSize << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Name\t\tCount" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (const auto& [name, count] : stats) {
        std::cout << name << "\t\t" << count << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

// =========================== Model Init ===========================
void initModels(seeta::FaceDetector*& detector,
    seeta::FaceLandmarker*& landmarker,
    seeta::FaceRecognizer*& recognizer) {
    detector = nullptr;
    landmarker = nullptr;
    recognizer = nullptr;

    try {
        // Detector
        seeta::ModelSetting detectorSetting;
        detectorSetting.append(g_detectorModel.c_str());
        detectorSetting.set_device(seeta::ModelSetting::CPU);
        detector = new seeta::FaceDetector(detectorSetting);

        // High precision settings
        detector->set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 20);
        detector->set(seeta::FaceDetector::PROPERTY_THRESHOLD, 0.90f);

        // Landmarker
        seeta::ModelSetting landmarkerSetting;
        landmarkerSetting.append(g_landmarkerModel.c_str());
        landmarker = new seeta::FaceLandmarker(landmarkerSetting);

        // Recognizer
        seeta::ModelSetting recognizerSetting;
        recognizerSetting.append(g_recognizerModel.c_str());
        recognizerSetting.set_device(seeta::ModelSetting::CPU);
        recognizer = new seeta::FaceRecognizer(recognizerSetting);

        // Get feature dimension
        g_featureSize = recognizer->GetExtractFeatureSize();
        std::cout << "Models loaded. Feature dimension: " << g_featureSize << std::endl;
    }
    catch (...) {
        delete detector;
        delete landmarker;
        delete recognizer;
        detector = nullptr;
        landmarker = nullptr;
        recognizer = nullptr;
        throw;
    }
}

// =========================== Menu ===========================
void printBanner() {
    std::cout << "|   Face Feature Management v1.0       |" << std::endl;
    std::cout << "Database: " << g_dbPath << " (" << g_database.size() << " records)" << std::endl;
}

void printMenu() {
    std::cout << "\nSelect operation:" << std::endl;
    std::cout << "  1. Batch extract face features (dataset root dir)" << std::endl;
    std::cout << "  2. Delete a person" << std::endl;
    std::cout << "  3. View database" << std::endl;
    std::cout << "  0. Exit" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Enter choice [0-3]: ";
}

// =========================== Main ===========================
int main(int argc, char* argv[]) {
    // Parse command line args
    if (argc >= 5) {
        g_detectorModel = argv[1];
        g_landmarkerModel = argv[2];
        g_recognizerModel = argv[3];
        g_dbPath = argv[4];
    }
    else if (argc >= 4) {
        g_detectorModel = argv[1];
        g_landmarkerModel = argv[2];
        g_recognizerModel = argv[3];
    }

    std::cout << "Model paths:" << std::endl;
    std::cout << "  Detector:  " << g_detectorModel << std::endl;
    std::cout << "  Landmarker:" << g_landmarkerModel << std::endl;
    std::cout << "  Recognizer:" << g_recognizerModel << std::endl;

    // Initialize SeetaFace6 models
    seeta::FaceDetector* detector = nullptr;
    seeta::FaceLandmarker* landmarker = nullptr;
    seeta::FaceRecognizer* recognizer = nullptr;

    try {
        initModels(detector, landmarker, recognizer);
    }
    catch (const std::exception& e) {
        std::cerr << "Model init error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Model init unknown error" << std::endl;
        return 1;
    }

    // Load database
    if (fs::exists(g_dbPath)) {
        if (!loadDatabase(g_dbPath)) {
            std::cerr << "Failed to load database, starting with empty database" << std::endl;
            g_database.clear();
        }
    }
    else {
        std::cout << "\nDatabase file not found: " << g_dbPath << std::endl;
        std::cout << "Create new database? (y/n): ";
        char answer;
        std::cin >> answer;
        std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
        if (answer == 'y' || answer == 'Y') {
            std::cout << "Creating new database..." << std::endl;
            g_database.clear();
            saveDatabase(g_dbPath);
        }
        else {
            std::cout << "Cancelled, exiting." << std::endl;
            delete detector;
            delete landmarker;
            delete recognizer;
            return 0;
        }
    }

    // Main loop
    bool running = true;
    printBanner();
    while (running) {
        printMenu();

        int choice = -1;
        std::cin >> choice;
        std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
            std::cout << "Invalid input, please enter a number" << std::endl;
            continue;
        }

        switch (choice) {
        case 1: {
            std::string dirPath;
            std::cout << "Enter dataset root directory: ";
            std::getline(std::cin, dirPath);
            if (!dirPath.empty()) {
                if (batchExtract(dirPath, *detector, *landmarker, *recognizer)) {
                    saveDatabase(g_dbPath);
                }
            }
            else {
                std::cout << "Input cannot be empty" << std::endl;
            }
            break;
        }
        case 2: {
            std::string name;
            std::cout << "Enter person name to delete: ";
            std::getline(std::cin, name);
            if (!name.empty()) {
                int removed = deletePerson(name);
                if (removed > 0) {
                    std::cout << "Deleted " << removed << " record(s) for " << name << std::endl;
                    saveDatabase(g_dbPath);
                }
                else {
                    std::cout << "No records found for \"" << name << "\"" << std::endl;
                }
            }
            break;
        }
        case 3: {
            viewDatabase();
            break;
        }
        case 0: {
            running = false;
            std::cout << "Saving database..." << std::endl;
            saveDatabase(g_dbPath);
            std::cout << "Goodbye!" << std::endl;
            break;
        }
        default: {
            std::cout << "Invalid option, please try again" << std::endl;
            break;
        }
        }
    }

    // Cleanup
    delete detector;
    delete landmarker;
    delete recognizer;

    return 0;
}