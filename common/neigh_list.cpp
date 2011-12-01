#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "unpickle.h"

const int NeighListLike::PGDELTA = 1;

NeighListLike::NeighListLike(struct params *input) {
  //allocate (resize maxlocal if necessary)
  allocate(input->nnode);

  //fill data arrays
  fill(input);

  //test
  test_against(input);
}

NeighListLike::~NeighListLike() {
  delete[] ilist;
  delete[] numneigh;
  delete[] firstneigh;
  delete[] firstdouble;
  delete[] firsttouch;
  for (int p=0; p<maxpage; p++) {
    delete[] pages[p];
    delete[] dpages[p];
    delete[] tpages[p];
  }
  // nb: we free these because we malloc them (because we realloc them)
  free(pages);
  free(dpages);
  free(tpages);
}

void NeighListLike::allocate(int N) {
  //default sizes
  maxlocal =  10000;
  maxpage  =      1;
  pgsize   = 100000;
  oneatom  =   2000;

  //give lengths to each list
  maxlocal = (N < maxlocal ? maxlocal : N);
  ilist = new int[maxlocal];
  assert(ilist);

  numneigh = new int[maxlocal];
  assert(numneigh);

  firstneigh = new int*[maxlocal];
  assert(firstneigh);
  firstdouble = new double*[maxlocal];
  assert(firstdouble);
  firsttouch = new int*[maxlocal];
  assert(firsttouch);

  pages = (int **)malloc(sizeof(int*)*maxpage);
  assert(pages);
  dpages = (double **)malloc(sizeof(double*)*maxpage);
  assert(dpages);
  tpages = (int **)malloc(sizeof(int*)*maxpage);
  assert(tpages);

  for (int p=0; p<maxpage; p++) {
    pages[p] = new int[pgsize];
    assert(pages[p]);
    dpages[p] = new double[pgsize*3];
    assert(dpages[p]);
    tpages[p] = new int[pgsize];
    assert(tpages[p]);
  }

  //initialize
  for (int i=0; i<maxlocal; i++) {
    ilist[i] = -1;
    numneigh[i] = 0;
    firstneigh[i] = NULL;
    firstdouble[i] = NULL;
    firsttouch[i] = NULL;
  }
  for (int p=0; p<maxpage; p++) {
    for (int i=0; i<pgsize; i++) {
      pages[p][i] = -1;
      dpages[p][(i*3)+0] = -1;
      dpages[p][(i*3)+1] = -1;
      dpages[p][(i*3)+2] = -1;
      tpages[p][i] = -1;
    }
  }
}

/* Increase the size of the neighbor list */
void NeighListLike::add_pages() {
  int npage = maxpage;
  maxpage += PGDELTA;
  pages = (int **)realloc(pages, maxpage*sizeof(int *));
  assert(pages);
  dpages = (double **)realloc(dpages, maxpage*sizeof(double *));
  assert(dpages);
  tpages = (int **)realloc(tpages, maxpage*sizeof(int *));
  assert(tpages);
  for (int p=npage; p<maxpage; p++) {
    pages[p] = new int[pgsize];
    assert(pages[p]);
    dpages[p] = new double[pgsize*3];
    assert(dpages[p]);
    tpages[p] = new int[pgsize];
    assert(tpages[p]);
  }
  for (int p=npage; p<maxpage; p++) {
    for (int i=0; i<pgsize; i++) {
      pages[p][i] = -1;
      dpages[p][(i*3)+0] = -1;
      dpages[p][(i*3)+1] = -1;
      dpages[p][(i*3)+2] = -1;
      tpages[p][i] = -1;
    }
  }
}

void NeighListLike::fill(struct params *input) {
  inum = input->nnode;
  for (int ii=0; ii<inum; ii++) {
    ilist[ii] = ii;
  }
  int p = 0;
  int npnt = 0;
  int lasti=0;
  for (int e=0; e<input->nedge; e++) {
    int i = input->edge[(e*2)  ];
    int j = input->edge[(e*2)+1];

    //index i must be monotonically increasing
    assert(lasti <= i); lasti = i;

    //move onto next page and add_pages if necessary
    if (pgsize - npnt < oneatom) {
      if (p == (maxpage-1)) {
        add_pages();
      }
      npnt = 0;
      p++;

      //fix-up if the current particle neighbors over two pages
      if (numneigh[i] > 0) {
        memcpy(&pages[p][0], firstneigh[i], numneigh[i]*sizeof(int));
        memcpy(&dpages[p][0], firstdouble[i], numneigh[i]*3*sizeof(double));
        memcpy(&tpages[p][0], firsttouch[i], numneigh[i]*sizeof(int));
        firstneigh[i] = &pages[p][0];
        firstdouble[i] = &dpages[p][0];
        firsttouch[i] = &tpages[p][0];
        npnt = numneigh[i];
      }
    }
    assert(npnt < pgsize);

    //fill pages and dpages
    if (numneigh[i] == 0) {
      firstneigh[i] = &pages[p][npnt];
      firstdouble[i] = &dpages[p][npnt*3];
      firsttouch[i] = &tpages[p][npnt];
    }
    pages[p][npnt] = j;
    dpages[p][(npnt*3)  ] = input->shear[(e*3)  ];
    dpages[p][(npnt*3)+1] = input->shear[(e*3)+1];
    dpages[p][(npnt*3)+2] = input->shear[(e*3)+2];
    tpages[p][npnt] = 1;
    numneigh[i]++;
    //check for overflow
    assert(numneigh[i] < oneatom);
    npnt++;
  }

  //fix-up neighbor list so there are no NULL values
  for (int ii=0; ii<inum; ii++) {
    int i = ilist[ii];
    if (firstneigh[i] == NULL) {
      if (i == 0) {
        firstneigh[i] = pages[0];
      } else {
        int j = i;
        do {
          j--;
          assert(j > 0);
        } while (firstneigh[j] == NULL);
        firstneigh[i] = firstneigh[j] + numneigh[j];
      }
    }
    assert(firstneigh[i] != NULL);
  }
}

void NeighListLike::test_against(struct params *input) {
  int nedge=0;
  for (int ii=0; ii<inum; ii++) {
    int i = ilist[ii];
    nedge += numneigh[i];
  }
  assert(nedge == input->nedge);

  int edge_ptr = 0;
  int *edge = new int[nedge*2];
  double *shear = new double[nedge*3];
  for (int ii=0; ii<inum; ii++) {
    int i = ilist[ii];
    for (int jj=0; jj<numneigh[i]; jj++) {
      int j = firstneigh[i][jj];
      edge[(edge_ptr*2)  ] = i;
      edge[(edge_ptr*2)+1] = j;
      double *s = &(firstdouble[i][3*jj]);
      shear[(edge_ptr*3)  ] = s[0];
      shear[(edge_ptr*3)+1] = s[1];
      shear[(edge_ptr*3)+2] = s[2];
      edge_ptr++;
      assert(firsttouch[i][jj] == 1);
    }
  }
  assert(edge_ptr == nedge);
  for (int e=0; e<input->nedge; e++) {
    assert(edge[(e*2)  ] == input->edge[(e*2)  ]);
    assert(edge[(e*2)+1] == input->edge[(e*2)+1]);
    assert(bitwise_equal(shear[(e*3)  ], input->shear[(e*3)  ]));
    assert(bitwise_equal(shear[(e*3)+1], input->shear[(e*3)+1]));
    assert(bitwise_equal(shear[(e*3)+2], input->shear[(e*3)+2]));
  }
  delete[] edge;
  delete[] shear;
}

void NeighListLike::copy_into(
  double **&firstdouble_copy,
  double **&dpages_copy,
  int    **&firsttouch_copy,
  int    **&tpages_copy) {
  //all null or initialized
  assert((firstdouble_copy == NULL && dpages_copy == NULL &&
          firsttouch_copy  == NULL && tpages_copy == NULL) ||
         (firstdouble_copy && dpages_copy &&
          firsttouch_copy && tpages_copy));

  if (firstdouble_copy == NULL && dpages_copy == NULL &&
      firsttouch_copy  == NULL && tpages_copy == NULL) {
    firstdouble_copy = new double*[maxlocal];
    assert(firstdouble_copy);
    firsttouch_copy = new int*[maxlocal];
    assert(firsttouch_copy);

    dpages_copy = new double*[maxpage];
    assert(dpages_copy);
    tpages_copy = new int*[maxpage];
    assert(tpages_copy);

    for (int p=0; p<maxpage; p++) {
      dpages_copy[p] = new double[pgsize*3];
      assert(dpages_copy[p]);
      tpages_copy[p] = new int[pgsize];
      assert(tpages_copy[p]);
    }
  }

  std::copy(firstdouble, firstdouble+maxlocal, firstdouble_copy);
  std::copy(firsttouch,  firsttouch+maxlocal,  firsttouch_copy);
  std::copy(dpages, dpages+maxpage, dpages_copy);
  std::copy(tpages, tpages+maxpage, tpages_copy);
  for (int p=0; p<maxpage; p++) {
    std::copy(dpages[p], dpages[p]+(pgsize*3), dpages_copy[p]);
    std::copy(tpages[p], tpages[p]+(pgsize  ), tpages_copy[p]);
  }
}

bool bitwise_equal(double const&d1, double const&d2) {
  union udouble {
    double d;
    unsigned long u;
  };

  udouble ud1; ud1.d = d1;
  udouble ud2; ud2.d = d2;
  return (ud1.u == ud2.u);
}

