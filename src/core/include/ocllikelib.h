#ifndef OCLLIKELIB_H
#define OCLLIKELIB_H

#include "gpuMuller.h"

class OCLlikeEval
{

private:
    bool contextSet;
    long siteCount, alphabetDimension;
    long* lNodeFlags;
    _SimpleList updateNodes,
                flatParents,
                flatNodes,
                flatCLeaves,
                flatLeaves,
                flatTree,
                theFrequencies;
    _Parameter  *iNodeCache,
                *theProbs;
    _SimpleList taggedInternals;
    _GrowingVector* lNodeResolutions;

    GPUMuller gm;
    double* treeCache;
    double* modelCache;

    void setupContext();
    double evaluate();
    double process_results(double* root_conditionals);

public:
    OCLlikeEval();
    OCLlikeEval(long siteCount, long alphabetDimension, _Parameter* iNodeCache);
    void init(long siteCount, long alphabetDimension, _Parameter* iNodeCache);
    double eval_likelihood( _SimpleList& updateNodes,
                            _SimpleList& flatParents,
                            _SimpleList& flatNodes,
                            _SimpleList& flatCLeaves,
                            _SimpleList& flatLeaves,
                            _SimpleList& flatTree,
                            _Parameter* theProbs,
                            _SimpleList& theFrequencies,
                            long* elNodeFlags,
                            _SimpleList& taggedInternals,
                            _GrowingVector* lNodeResolutions);
    ~OCLlikeEval();
};


#endif
