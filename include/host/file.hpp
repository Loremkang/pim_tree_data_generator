#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>

template <typename T, bool check_sorted_deduplicated = false>
std::vector<T> LoadElementsFromBinary(const std::string& file_path,
                                  const uint64_t start, const uint64_t end) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    } else {
        std::cout << "Open File: " << file_path << std::endl;
    }

    uint64_t element_count;
    uint64_t element_size;

    file.read(reinterpret_cast<char*>(&element_count), sizeof(uint64_t));
    std::cout << "File has " << element_count << " elements" << std::endl;

    assert(end > start);

    uint64_t read_count = std::min(element_count, end - start);
    std::cout << "Read " << read_count << " elements from " << start << " to "
              << end << std::endl;

    file.read(reinterpret_cast<char*>(&element_size), sizeof(uint64_t));
    std::cout << "Each element has " << element_size << " bytes" << std::endl;
    assert(element_size == sizeof(T));

    uint64_t read_size = read_count * element_size;
    std::cout << "Read " << read_size << " bytes" << std::endl;

    std::vector<T> data(read_count);

    file.seekg(sizeof(uint64_t) * 2 + start * element_size, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), read_size);

    if (check_sorted_deduplicated) {
        parlay::parallel_for(1, read_count,
                             [&](size_t i) { assert(data[i] > data[i - 1]); });
    }

    return data;
}

template <typename T>
void StoreElementsToBinary(const std::string& file_path,
                           const std::vector<T>& data) {
    if (std::ifstream(file_path)) {
        throw std::runtime_error("File already exists: " + file_path);
    }

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    uint64_t element_count = data.size();
    uint64_t element_size = sizeof(T);
    uint64_t total_size = element_count * element_size;

    file.write(reinterpret_cast<const char*>(&element_count), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&element_size), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(data.data()), total_size);
}