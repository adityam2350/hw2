NVCC      := nvcc
NVCCFLAGS := -O2

BINDIR := bin
OBJDIR := bin/obj

# ----- Tile-size overrides for the conv kernel -----
# Default values match the in-source #defines in conv.cu.
# CHUNK_NI is fixed at 64 for this assignment; only TILE_X/TILE_Y select
# the tiled binary name. Filter size Ky/Kx and channels Ni/Nn are runtime
# CLI flags on ./bin/conv (see main.cpp).
# Override on the command line, e.g.:
#   make tiled TILE_X=10 TILE_Y=10 CHUNK_NI=64
TILE_X   ?= 8
TILE_Y   ?= 8
CHUNK_NI ?= 64

TILEFLAGS := -DTILE_X=$(TILE_X) -DTILE_Y=$(TILE_Y) -DCHUNK_NI=$(CHUNK_NI)
TILE_TAG  := t$(TILE_X)_$(TILE_Y)_c$(CHUNK_NI)

TILED_BIN := $(BINDIR)/conv_$(TILE_TAG)

all: $(BINDIR)/conv

# ----- Default build (tile sizes from conv.cu) -----
$(OBJDIR)/conv.o: conv.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJDIR)/main.o: main.cpp | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BINDIR)/conv: $(OBJDIR)/main.o $(OBJDIR)/conv.o | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# ----- Tiled build (tile sizes from TILE_X/TILE_Y/CHUNK_NI) -----
# Per-tile object directory so different tile configs don't clobber each other.
$(OBJDIR)/$(TILE_TAG)/conv.o: conv.cu | $(OBJDIR)/$(TILE_TAG)
	$(NVCC) $(NVCCFLAGS) $(TILEFLAGS) -c -o $@ $<

$(OBJDIR)/$(TILE_TAG)/main.o: main.cpp | $(OBJDIR)/$(TILE_TAG)
	$(NVCC) $(NVCCFLAGS) $(TILEFLAGS) -c -o $@ $<

$(TILED_BIN): $(OBJDIR)/$(TILE_TAG)/main.o $(OBJDIR)/$(TILE_TAG)/conv.o | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

.PHONY: tiled
tiled: $(TILED_BIN)
	@echo "Built $(TILED_BIN) (TILE_X=$(TILE_X) TILE_Y=$(TILE_Y) CHUNK_NI=$(CHUNK_NI))"

$(BINDIR) $(OBJDIR):
	mkdir -p $@

$(OBJDIR)/$(TILE_TAG):
	mkdir -p $@

NCU      := ncu
NCUFLAGS := --set full
NCUTMP   := .ncu-tmp

.PHONY: profile-conv clean
profile-conv: $(BINDIR)/conv
	@mkdir -p $(NCUTMP)
	TMPDIR=$(NCUTMP) $(NCU) $(NCUFLAGS) $(BINDIR)/conv

clean:
	rm -rf $(BINDIR) $(NCUTMP)
