# Make file for Proximal Bundle Method
# put your compile flags here
FF = gfortran

all:	mpbngc.dylib

tmpbngc.o: tmpbngc.f
	$(FF) -c tmpbngc.f

mpbngc.o: mpbngc.f
	$(FF) -c mpbngc.f

plqdf1.o: plqdf1.f
	$(FF) -c plqdf1.f

pllpb2.o: pllpb2.f
	$(FF) -c pllpb2.f

mpbngc.dylib: tmpbngc.o mpbngc.o plqdf1.o pllpb2.o
	$(FF) -dynamiclib -o mpbngc.dylib tmpbngc.o mpbngc.o plqdf1.o pllpb2.o

