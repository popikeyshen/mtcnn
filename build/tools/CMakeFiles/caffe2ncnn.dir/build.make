# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/popikeyshen/mtcnn-ncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/popikeyshen/mtcnn-ncnn/build

# Include any dependencies generated for this target.
include tools/CMakeFiles/caffe2ncnn.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/caffe2ncnn.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/caffe2ncnn.dir/flags.make

tools/caffe.pb.cc: ../tools/caffe.proto
tools/caffe.pb.cc: /usr/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/popikeyshen/mtcnn-ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running C++ protocol buffer compiler on caffe.proto"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/protoc --cpp_out=/home/popikeyshen/mtcnn-ncnn/build/tools -I /home/popikeyshen/mtcnn-ncnn/tools /home/popikeyshen/mtcnn-ncnn/tools/caffe.proto

tools/caffe.pb.h: tools/caffe.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate tools/caffe.pb.h

tools/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: tools/CMakeFiles/caffe2ncnn.dir/flags.make
tools/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o: ../tools/caffe2ncnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/popikeyshen/mtcnn-ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tools/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o -c /home/popikeyshen/mtcnn-ncnn/tools/caffe2ncnn.cpp

tools/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/popikeyshen/mtcnn-ncnn/tools/caffe2ncnn.cpp > CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.i

tools/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/popikeyshen/mtcnn-ncnn/tools/caffe2ncnn.cpp -o CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.s

tools/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/CMakeFiles/caffe2ncnn.dir/flags.make
tools/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o: tools/caffe.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/popikeyshen/mtcnn-ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tools/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o -c /home/popikeyshen/mtcnn-ncnn/build/tools/caffe.pb.cc

tools/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/popikeyshen/mtcnn-ncnn/build/tools/caffe.pb.cc > CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.i

tools/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/popikeyshen/mtcnn-ncnn/build/tools/caffe.pb.cc -o CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.s

# Object files for target caffe2ncnn
caffe2ncnn_OBJECTS = \
"CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o" \
"CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o"

# External object files for target caffe2ncnn
caffe2ncnn_EXTERNAL_OBJECTS =

tools/caffe2ncnn: tools/CMakeFiles/caffe2ncnn.dir/caffe2ncnn.cpp.o
tools/caffe2ncnn: tools/CMakeFiles/caffe2ncnn.dir/caffe.pb.cc.o
tools/caffe2ncnn: tools/CMakeFiles/caffe2ncnn.dir/build.make
tools/caffe2ncnn: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/caffe2ncnn: tools/CMakeFiles/caffe2ncnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/popikeyshen/mtcnn-ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable caffe2ncnn"
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe2ncnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/caffe2ncnn.dir/build: tools/caffe2ncnn

.PHONY : tools/CMakeFiles/caffe2ncnn.dir/build

tools/CMakeFiles/caffe2ncnn.dir/clean:
	cd /home/popikeyshen/mtcnn-ncnn/build/tools && $(CMAKE_COMMAND) -P CMakeFiles/caffe2ncnn.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/caffe2ncnn.dir/clean

tools/CMakeFiles/caffe2ncnn.dir/depend: tools/caffe.pb.cc
tools/CMakeFiles/caffe2ncnn.dir/depend: tools/caffe.pb.h
	cd /home/popikeyshen/mtcnn-ncnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/popikeyshen/mtcnn-ncnn /home/popikeyshen/mtcnn-ncnn/tools /home/popikeyshen/mtcnn-ncnn/build /home/popikeyshen/mtcnn-ncnn/build/tools /home/popikeyshen/mtcnn-ncnn/build/tools/CMakeFiles/caffe2ncnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/caffe2ncnn.dir/depend

