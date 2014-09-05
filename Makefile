LIB	:= libnn.so
VER	:= 1

CXX				:= g++
CXXFLAGS	:= -O3 -Wall
INCLUDE		:= -I../utils/DataStructures
LDFLAGS		:= -shared -Wl,-soname,$(LIB).$(VER)

PIC	:= -fPIC

SRC	:= $(shell ls *.cpp)
HDR	:= $(shell ls *.h)
OBJ	:= $(SRC:.cpp=.o)

all: $(LIB)

clean:
	@echo " CLEAN $(LIB) $(OBJ)"
	@rm -rf *.o $(LIB) $(LIB).$(VER) $(OBJ)

%.o: %.cpp $(HDR)
	@echo " CXX $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDE) $(PIC) -c $<

$(LIB): $(OBJ)
	@echo " LD  $(LIB)"
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(LIB).$(VER) $(OBJ)
	@ln -fs ./$(LIB).$(VER) ./$(LIB)

.PHONY: all clean
