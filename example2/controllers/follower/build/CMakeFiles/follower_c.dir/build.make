# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.21.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.21.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build

# Include any dependencies generated for this target.
include CMakeFiles/follower_c.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/follower_c.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/follower_c.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/follower_c.dir/flags.make

CMakeFiles/follower_c.dir/follower_c.c.o: CMakeFiles/follower_c.dir/flags.make
CMakeFiles/follower_c.dir/follower_c.c.o: ../follower_c.c
CMakeFiles/follower_c.dir/follower_c.c.o: CMakeFiles/follower_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/follower_c.dir/follower_c.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/follower_c.dir/follower_c.c.o -MF CMakeFiles/follower_c.dir/follower_c.c.o.d -o CMakeFiles/follower_c.dir/follower_c.c.o -c /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/follower_c.c

CMakeFiles/follower_c.dir/follower_c.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/follower_c.dir/follower_c.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/follower_c.c > CMakeFiles/follower_c.dir/follower_c.c.i

CMakeFiles/follower_c.dir/follower_c.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/follower_c.dir/follower_c.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/follower_c.c -o CMakeFiles/follower_c.dir/follower_c.c.s

# Object files for target follower_c
follower_c_OBJECTS = \
"CMakeFiles/follower_c.dir/follower_c.c.o"

# External object files for target follower_c
follower_c_EXTERNAL_OBJECTS =

follower_c: CMakeFiles/follower_c.dir/follower_c.c.o
follower_c: CMakeFiles/follower_c.dir/build.make
follower_c: CMakeFiles/follower_c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable follower_c"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/follower_c.dir/link.txt --verbose=$(VERBOSE)
	/usr/local/Cellar/cmake/3.21.2/bin/cmake -E copy /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build/follower_c /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c

# Rule to build all files generated by this target.
CMakeFiles/follower_c.dir/build: follower_c
.PHONY : CMakeFiles/follower_c.dir/build

CMakeFiles/follower_c.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/follower_c.dir/cmake_clean.cmake
.PHONY : CMakeFiles/follower_c.dir/clean

CMakeFiles/follower_c.dir/depend:
	cd /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build /Users/beyazit/Documents/Scenic/examples/webots/generic/controllers/follower_c/build/CMakeFiles/follower_c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/follower_c.dir/depend

