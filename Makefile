all:
	cd common; make
	cd serial; make
	cd cuda; make

clean:
	cd common; make clean
	cd serial; make clean
	cd cuda; make clean
