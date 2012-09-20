/*

 HyPhy - Hypothesis Testing Using Phylogenies.

 Copyright (C) 1997-now
 Core Developers:
 Sergei L Kosakovsky Pond (spond@ucsd.edu)
 Art FY Poon    (apoon@cfenet.ubc.ca)
 Steven Weaver (sweaver@ucsd.edu)

 Module Developers:
 Lance Hepler (nlhepler@gmail.com)
 Martin Smith (martin.audacis@gmail.com)

 Significant contributions from:
 Spencer V Muse (muse@stat.ncsu.edu)
 Simon DW Frost (sdf22@cam.ac.uk)

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 */

#ifdef MDSOCL

#include "calcnode.h"
#include <math.h>
#include "ocllikelib.h"
#include <iostream>

using namespace std;

OCLlikeEval::OCLlikeEval(   long siteCount,
                            long alphabetDimension,
                            _Parameter* iNodeCache)
{
    init(siteCount, alphabetDimension, iNodeCache);
}

OCLlikeEval::OCLlikeEval()
{
    // Do nothing
}

void OCLlikeEval::init( long siteCount,
                        long alphabetDimension,
                        _Parameter* iNodeCache)
{
    this->siteCount = siteCount;
    this->alphabetDimension = alphabetDimension;
    this->iNodeCache = iNodeCache;
    gm = GPUMuller();
    contextSet = false;
    //cout << "OCLLikeEval initialized!" << endl;
}


void OCLlikeEval::setupContext()
{
    // XXX make this compatible with ambiguous nodes
    long treesize = alphabetDimension * siteCount * (flatNodes.lLength +
    flatLeaves.lLength);
    treeCache = new double[ alphabetDimension
                            * siteCount
                            * (flatNodes.lLength
                            + flatLeaves.lLength)];

    int leafThresh = alphabetDimension * siteCount * flatLeaves.lLength;
    for (int n = 0; n < flatLeaves.lLength; n++)
    {
        for (int s = 0; s < siteCount; s++)
        {
            for (int c = 0; c < alphabetDimension; c++)
            {
                treeCache[  n * siteCount * alphabetDimension
                            + s * alphabetDimension + c] = 0.0;
            }
        }
    }

    for (long n = 0; n < updateNodes.lLength; n++)
    {
        long nodeCode = updateNodes.lData[n];
        long parentCode = flatParents.lData[nodeCode];
        bool isLeaf = nodeCode < flatLeaves.lLength;
        if (isLeaf)
        {
            //cout << "NodeCode: " << nodeCode << endl;
            for (long s = 0; s < siteCount; s++)
            {
                long siteState = lNodeFlags[nodeCode*siteCount + s];
                if (siteState >= 0)
                {
                    treeCache[  nodeCode
                                * siteCount
                                * alphabetDimension
                                + s
                                * alphabetDimension
                                + siteState] = 1.0;
                                /*
                    cout    << "filling: "
                            <<  nodeCode
                                * siteCount
                                * alphabetDimension
                                + s
                                * alphabetDimension
                                + siteState
                            << endl;
                            */
                }
                else
                    cout << "no resolution?" << endl;
            }
        }
        // XXX ELSE resolve ambiguous nodes!
    }


    modelCache = new double[alphabetDimension
                            * alphabetDimension
                            * flatParents.lLength - 1];

    int ANR = (flatNodes.lLength + flatLeaves.lLength) * siteCount;
    int ANC = alphabetDimension;
    gm.set_A(treeCache, ANR, ANC);

    int BNR = (flatParents.lLength -1) * alphabetDimension;
    int BNC = alphabetDimension;
    gm.set_B(modelCache, BNR, BNC);

    gm.set_C(treeCache, ANR, ANC);
}

double OCLlikeEval::evaluate()
{
    for (int n = 0; n < updateNodes.lLength; n++)
    {
        int nodeCode = updateNodes.lData[n];
        int parentCode = flatParents.lData[n];
        bool isLeaf = nodeCode < flatLeaves.lLength;
        if (!isLeaf) nodeCode -= flatLeaves.lLength;

        _Parameter* tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                    ((_CalcNode*) flatTree
                    (nodeCode)))->GetCompExp(0)->theData;

        for (int pc = 0; pc < alphabetDimension; pc++)
        {
            for (int cc = 0; cc < alphabetDimension; cc++)
            {
                modelCache[nodeCode * alphabetDimension * alphabetDimension + pc *
                alphabetDimension + cc] = (double)(tMatrix[pc *
                alphabetDimension + cc]);
            }
        }
    }
    int BNR = (flatParents.lLength - 1) * alphabetDimension;
    int BNC = alphabetDimension;
    gm.update_B(modelCache, 0, 0, BNR, BNC, BNR, BNC);

    int ARO, ACO, AH, UD, BW, BRO, BCO, CRO, CCO;

    for (int n = 0; n < updateNodes.lLength; n++)
    {
        long nodeCode = updateNodes.lData[n];
        long parentCode = flatParents.lData[nodeCode];

        cout << "nodeCode: " << nodeCode << endl;
        cout << "parentCode: " << parentCode << endl;

        // At this point you would normally subtract the number of leaves
        // from nodeCode, but as we are recreating the whole tree it is fine
        // if we leave nodeCode whole
        ARO = nodeCode * siteCount;
        ACO = 0;
        AH = siteCount;
        UD = alphabetDimension;


/*
*/
        cout    << "ARO, ACO, AH, UD: "
                << ARO
                << ", "
                << ACO
                << ", "
                << AH
                << ", "
                << UD
                << endl;
        gm.bound_A(ARO, ACO, AH, UD);

        BRO = nodeCode * alphabetDimension;
        BCO = 0;
        BW = alphabetDimension;

/*
*/
        cout    << "BRO, BCO, UD, BW: "
                << BRO
                << ", "
                << BCO
                << ", "
                << UD
                << ", "
                << BW
                << endl;
        gm.bound_B(BRO, BCO, UD, BW);

        // However just as we would normally subtract the number of leaves
        // from nodeCode but we don't, we now have to add the number of
        // leaves to parentCode so that stuff is saved in the correct place
        CRO = (parentCode + flatLeaves.lLength) * siteCount;
        CCO = 0;

/*
*/
        cout    << "CRO, CCO: "
                << CRO
                << ", "
                << CCO
                << endl;

        if (taggedInternals.lData[parentCode] == 0)
        {
            taggedInternals.lData[parentCode] = 1;
            gm.set_overwrite(true);
            //cout << "overwrite!" << endl;
        }
        else
        {
            gm.set_overwrite(false);
            //cout << "don't overwrite!" << endl;
        }

        gm.eval_C(CRO, CCO, AH, BW);
    }
    int rootRO = (flatLeaves.lLength + flatNodes.lLength - 1) * siteCount;
    int rootCO = 0;
    int rootH = siteCount;
    int rootW = alphabetDimension;
    // XXX check to make sure there is an option for slicing without mangling
    // memory
    cout    << "rootRO, rootCO: "
            << rootRO
            << ", "
            << rootCO
            << ", "
            << "rootH, rootW: "
            << rootH
            << ", "
            << rootW
            << endl;
    double* result = gm.get_C_double(rootRO, rootCO, rootH, rootW);

    return process_results(result);
}

double OCLlikeEval::process_results(double* root_conditionals)
{
    cout << "Root Conditionals: " << endl;
    double result = 0.0;
    long root_index = 0;
    for (long s = 0; s < siteCount; s++)
    {
        double accumulator = 0.0;
        for (long pc = 0; pc < alphabetDimension; pc++)
        {
            cout << root_conditionals[root_index] << " ";
            accumulator += root_conditionals[root_index] * theProbs[pc];
            root_index++;
        }
        cout << endl;
        result += log(accumulator) * theFrequencies[s];
    }
    return result;
}

double OCLlikeEval::eval_likelihood(    _SimpleList& updateNodes,
                                        _SimpleList& flatParents,
                                        _SimpleList& flatNodes,
                                        _SimpleList& flatCLeaves,
                                        _SimpleList& flatLeaves,
                                        _SimpleList& flatTree,
                                        _Parameter* theProbs,
                                        _SimpleList& theFrequencies,
                                        long* lNodeFlags,
                                        _SimpleList& taggedInternals,
                                        _GrowingVector* lNodeResolutions)
{
    this->updateNodes = updateNodes;
    this->taggedInternals = taggedInternals;
    this->theFrequencies = theFrequencies;
    if (!contextSet)
    {
        this->theProbs = theProbs;
        this->flatNodes = flatNodes;
        this->flatCLeaves = flatCLeaves;
        this->flatLeaves = flatLeaves;
        this->flatTree = flatTree;
        this->flatParents = flatParents;
        this->lNodeFlags = lNodeFlags;
        this->lNodeResolutions = lNodeResolutions;
        setupContext();
        contextSet = true;
    }

    return evaluate();
}


OCLlikeEval::~OCLlikeEval()
{
    int i = 0;
}


#endif
