cmake_minimum_required(VERSION 3.10.0)
project(p2 VERSION 0.1.0 LANGUAGES C CXX)

if (MSVC)
    add_compile_options(/W3)
    add_compile_options(/std:c++23preview)
    add_compile_options(/EHsc)
    add_compile_options(/utf-8)
    add_compile_options(/Zc:tlsGuards)
    add_compile_options(/Zc:threadSafeInit)
    add_compile_options(/O3)
    
else()
    add_compile_options(-Wall)
    add_compile_options(-std=c++20)
    add_compile_options(-fexceptions)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    add_compile_options(-O3)
endif()


add_executable(p2
    main.cpp 
    random.cpp
)

