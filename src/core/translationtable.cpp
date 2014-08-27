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
// XXX remove lData references

#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "hy_strings.h"
#include "hy_globals.h"
#include "translationtable.h"
#include "site.h"
#include "errorfns.h"

_TranslationTable default_translation_table;

_String amino_acid_one_char_codes("ACDEFGHIKLMNPQRSTVWY"), dna_one_char_codes("ACGT"),
    rna_one_char_codes("ACGU"), binary_one_char_codes("01");

//______________________________________________________________________________
_TranslationTable::_TranslationTable(void) {
  base_length = 4;
  check_table = NULL;
}

//______________________________________________________________________________
_TranslationTable::_TranslationTable(_TranslationTable &t) {
  this->Duplicate(&t);
}

//______________________________________________________________________________
_TranslationTable::_TranslationTable(const _String &alphabet) {
  base_length = alphabet.s_length;
  check_table = NULL;
  if (!(alphabet.Equal(&dna_one_char_codes) ||
        alphabet.Equal(&rna_one_char_codes) ||
        alphabet.Equal(&binary_one_char_codes) ||
        alphabet.Equal(&amino_acid_one_char_codes))) {
    this->addBaseSet(alphabet);
  }
}

//______________________________________________________________________________
void _TranslationTable::Duplicate(BaseRefConst obj) {
  _TranslationTable *tt = (_TranslationTable *)obj;
  base_length = tt->base_length;
  check_table = nil;
  tokens_added = tt->tokens_added;
  base_set = tt->base_set;
  translations_added.Duplicate(&tt->translations_added);
}

//______________________________________________________________________________
BaseRef _TranslationTable::makeDynamic(void) const{
  _TranslationTable *r = new _TranslationTable;
  r->Duplicate(this);
  return r;
}

//______________________________________________________________________________
const unsigned long _TranslationTable::tokenCode(const char token) const {
  // standard translations
  long receptacle[HY_WIDTH_OF_LONG];
  this->tokenCode(token, receptacle);
  return bitStringToLong(receptacle, base_length);
}

//______________________________________________________________________________
// assumes a non-unique translation of split
// for unique - use convertCodeToLetters
char _TranslationTable::codeToLetter(long *split) const {
  const unsigned long trsl = bitStringToLong( split,
                                              this->lengthOfAlphabet());

  if (base_set.s_length == 0) {
    // one of the standard alphabers
    if (base_length == 4) {
      // nucleotides
      switch (trsl) {
      case 3:
        return 'M';
      case 5:
        return 'S';
      case 6:
        return 'R';
      case 7:
        return 'V';
      case 9:
        return 'W';
      case 10:
        return 'Y';
      case 11:
        return 'H';
      case 12:
        return 'K';
      case 14:
        return 'B';
      }
    } else if (base_length == 20) {
      // amino acids
      switch (trsl) {
      case 2052:
        return 'B';
      case 8200:
        return 'Z';
      }
    }
  } else if (tokens_added.s_length) {
    long f = translations_added.Find(trsl);
    // linear search for (binary) translations
    if (f >= 0) {
      return tokens_added.s_data[f];
    }
  }
  return '?';
}

//______________________________________________________________________________
void _TranslationTable::splitTokenCode(const long code,
                                       long *receptacle) const {
  longToBitString(receptacle, code, base_length);
}

//______________________________________________________________________________
const unsigned long _TranslationTable::lengthOfAlphabet(void) const {
  return base_set.s_length ? base_set.s_length : base_length;
}

//______________________________________________________________________________

bool _TranslationTable::tokenCode(const char token, long *receptacle,
                                  const bool gap_to_ones) const {

  long f = tokens_added.s_length ? tokens_added.Find(token) : HY_NOT_FOUND;
  // check for custom translations
  // OPTIMIZE FLAG linear search:
  // SLKP 20071002 should really be a 256 char lookup table

  if (f != HY_NOT_FOUND) {
    this->splitTokenCode(translations_added(f), receptacle);
    return true;
  }

  if (base_set.s_length) {
    // custom base alphabet

    memset(receptacle, 0, base_length * sizeof(long));
    f = base_set.Find(token);
    // OPTIMIZE FLAG linear search:
    // SLKP 20071002 should really be a 256 char lookup table

    if (f != HY_NOT_FOUND) {
      receptacle[f] = 1;
    }

    return true;
  }

  if (base_length == 4) {
    // standard nucleotide
    memset(receptacle, 0, 4L * sizeof(long));

    switch (token) {
    case 'A':
      receptacle[0] = 1;
      break;

    case 'C':
      receptacle[1] = 1;
      break;

    case 'G':
      receptacle[2] = 1;
      break;

    case 'T':
    case 'U':
      receptacle[3] = 1;
      break;

    case 'Y':
      receptacle[3] = 1;
      receptacle[1] = 1;
      break;

    case 'R':
      receptacle[0] = 1;
      receptacle[2] = 1;
      break;

    case 'W':
      receptacle[3] = 1;
      receptacle[0] = 1;
      break;

    case 'S':
      receptacle[1] = 1;
      receptacle[2] = 1;
      break;

    case 'K':
      receptacle[3] = 1;
      receptacle[2] = 1;
      break;

    case 'M':
      receptacle[1] = 1;
      receptacle[0] = 1;
      break;

    case 'B':
      receptacle[1] = 1;
      receptacle[2] = 1;
      receptacle[3] = 1;
      break;

    case 'D':
      receptacle[0] = 1;
      receptacle[2] = 1;
      receptacle[3] = 1;
      break;

    case 'H':
      receptacle[1] = 1;
      receptacle[0] = 1;
      receptacle[3] = 1;
      break;

    case 'V':
      receptacle[1] = 1;
      receptacle[2] = 1;
      receptacle[0] = 1;
      break;

    case 'X':
    case 'N':
    case '?':
    case '.':
    case '*':
      receptacle[1] = 1;
      receptacle[2] = 1;
      receptacle[3] = 1;
      receptacle[0] = 1;
      break;

    case '-':
      if (gap_to_ones) {
        receptacle[1] = 1;
        receptacle[2] = 1;
        receptacle[3] = 1;
        receptacle[0] = 1;
        break;
      }
    }
  } else {
    if (base_length == 20) {
      memset(receptacle, 0, 20L * sizeof(long));

      switch (token) {
      case 'A':
        receptacle[0] = 1;
        break;

      case 'B':
        receptacle[2] = 1;
        receptacle[11] = 1;
        break;

      case 'C':
        receptacle[1] = 1;
        break;

      case 'D':
        receptacle[2] = 1;
        break;

      case 'E':
        receptacle[3] = 1;
        break;

      case 'F':
        receptacle[4] = 1;
        break;

      case 'G':
        receptacle[5] = 1;
        break;

      case 'H':
        receptacle[6] = 1;
        break;

      case 'I':
        receptacle[7] = 1;
        break;

      case 'K':
        receptacle[8] = 1;
        break;

      case 'L':
        receptacle[9] = 1;
        break;

      case 'M':
        receptacle[10] = 1;
        break;

      case 'N':
        receptacle[11] = 1;
        break;

      case 'P':
        receptacle[12] = 1;
        break;

      case 'Q':
        receptacle[13] = 1;
        break;

      case 'R':
        receptacle[14] = 1;
        break;

      case 'S':
        receptacle[15] = 1;
        break;

      case 'T':
        receptacle[16] = 1;
        break;

      case 'V':
        receptacle[17] = 1;
        break;

      case 'W':
        receptacle[18] = 1;
        break;

      case 'Y':
        receptacle[19] = 1;
        break;

      case 'Z':
        receptacle[3] = 1;
        receptacle[13] = 1;
        break;

      case 'X':
      case '?':
      case '.':
      case '*': {
        for (int j = 0; j < 20; j++) {
          receptacle[j] = 1;
        }
      } break;
      case '-': {
        if (gap_to_ones)
          for (int j = 0; j < 20; j++) {
            receptacle[j] = 1;
          }
      } break;
      }
    } else {
      // binary
      receptacle[0] = 0;
      receptacle[1] = 0;
      switch (token) {
      case '0':
        receptacle[0] = 1;
        break;

      case '1':
        receptacle[1] = 1;
        break;

      case 'X':
      case '?':
      case '.':
      case '*': {
        receptacle[0] = 1;
        receptacle[1] = 1;
      } break;
      case '-': {
        if (gap_to_ones) {
          receptacle[0] = 1;
          receptacle[1] = 1;
        }
      } break;
      }

    }
  }
  return false;

}

//______________________________________________________________________________
void _TranslationTable::prepareForChecks(void) {
  if (check_table == NULL) {
    check_table = new unsigned char [256];
  }

  memset(check_table, 0, 256);

  _String checkSymbols;

  if (base_set.s_length) {
    checkSymbols = base_set & tokens_added;
  } else if (base_length == 2) {
    checkSymbols = _String("01*?-.") & tokens_added;
  } else {
    checkSymbols = _String("ABCDEFGHIJKLMNOPQRSTUVWXYZ*?-.") & tokens_added;
  }

  for (unsigned long i = 0; i < checkSymbols.s_length; i++) {
    check_table[checkSymbols.getChar(i)] = 1;
  }
}

//______________________________________________________________________________
const bool _TranslationTable::isCharLegal(const char c) {
  if (!check_table) {
    this->prepareForChecks();
  }
  return check_table[c];
}

//______________________________________________________________________________
void _TranslationTable::addTokenCode(const char token, _String &code) {
  long f, newCode = 0L;

  bool reset_baseset = false;

  if (base_set.s_length == 0) {
    // fill in base_set for standard alphabets
    if (base_length == 4) {
      base_set = dna_one_char_codes;
    } else if (base_length == 20) {
      base_set = amino_acid_one_char_codes;
    } else {
      base_set = binary_one_char_codes;
    }
    reset_baseset = true;
  }

  if (base_set.s_length) {
    long shifter = 1;
    for (unsigned long j = 0; j < base_set.s_length; j++, shifter <<= 1) {
      if (code.Find(base_set.s_data[j]) != HY_NOT_FOUND) {
        newCode += shifter;
      }
    }
  }

  f = base_set.Find(token);
  if (reset_baseset) {
    base_set = empty;
  }

  if (f != HY_NOT_FOUND) {
    return;
  }
  // see if the character being added is a base
  // character; those cannot be redefined

  f = tokens_added.Find(token, 0, -1);
  // new definition or redefinition?

  if (f == HY_NOT_FOUND) { // new
    tokens_added = tokens_added & token;
    translations_added << 0L;
    f = tokens_added.s_length - 1L;
  }

  //translations_added.lData[f] = newCode;
  translations_added[f] = newCode;
}

//______________________________________________________________________________
void _TranslationTable::addBaseSet(const _String &code) {
  base_set = code;
  base_set.StripQuotes();
  if (this->checkValidAlphabet(code)) {
    base_length = base_set.s_length;
    if (base_length > HY_WIDTH_OF_LONG) {
      // longer than the bit size of 'long'
      // can't handle those
      warnError(_String("Alphabets with more than ") & HY_WIDTH_OF_LONG &
                " characters are not supported");
    }
  }
}

//______________________________________________________________________________
const char _TranslationTable::getSkipChar(void) const {
  if (this->detectType() != HY_TRANSLATION_TABLE_ANY_STANDARD) {
    return '?'; // this is the default
  }

  // see if there is a symbol
  // which maps to all '1'

  long all = 0, shifter = 1;

  unsigned long ul = this->lengthOfAlphabet();

  for (unsigned long f = 0; f < ul; f++, shifter <<= 1) {
    all |= shifter;
  }

  if ((all = translations_added.Find(all)) == HY_NOT_FOUND) {
    return '?';
  } else {
    return tokens_added.getChar(all);
  }
}

//______________________________________________________________________________
const char _TranslationTable::getGapChar(void) const {
  if (this->detectType() != HY_TRANSLATION_TABLE_ANY_STANDARD) {
    return '-'; // default gap character
  }

  long f = translations_added.Find(0L);

  if (f == HY_NOT_FOUND) {
    return 0;
  } else {
    return tokens_added.getChar(f);
  }
}

//______________________________________________________________________________
_String _TranslationTable::convertCodeToLetters(long code, const char base) {

  _String res(base, false);

  const unsigned long ul = this->lengthOfAlphabet();

  if (code >= 0) {
    // OPTIMIZE FLAG; repeated memory allocation/deallocation
    if (base_set.s_length) {
      for (long k = 1; k <= base; k++, code /= ul) {
        res.s_data[base - k] = base_set.s_data[code % ul];
      }
    } else {
      const _String *std_alphabet =
          _TranslationTable::getDefaultAlphabet(base_length);
      if (std_alphabet) {
        for (long k = 1; k <= base; k++, code /= ul) {
          res.s_data[base - k] = std_alphabet->getChar(code % ul);
        }
      } else {
        warnError("Internal error in _TranslationTable::convertCodeToLetters; "
                  "unsupported standard alphabet");
      }
    }

  } else {
    char c = this->getGapChar();
    for (long k = 0; k < base; k++) {
      res.s_data[k] = c;
    }
  }
  return res;
}

//______________________________________________________________________________
void _TranslationTable::clear(void) {
  base_length = 4;
  base_set = empty;
  tokens_added = empty;
  translations_added.Clear();
  if (check_table) {
    free(check_table);
    check_table = nil;
  }
}

//______________________________________________________________________________
void _TranslationTable::setStandardType(unsigned const char type) {
  this->clear();
  switch (type) {
  case HY_TRANSLATION_TABLE_STANDARD_BINARY:
    base_length = 2;
    return;
  case HY_TRANSLATION_TABLE_STANDARD_PROTEIN:
    base_length = 20;
    return;
  }
  base_length = 4;
}

//______________________________________________________________________________
bool _TranslationTable::checkType(unsigned char pattern) const {
  return this->detectType() & pattern;
}

//______________________________________________________________________________
const unsigned char _TranslationTable::detectType(void) const {
  //if (base_set.s_length == 0UL && translations_added.lLength == 0UL &&
  if (base_set.s_length == 0UL && translations_added.countitems()== 0UL &&
      tokens_added.s_length == 0UL) {
    switch (base_length) {
    case 2:
      return HY_TRANSLATION_TABLE_STANDARD_BINARY;
    case 4:
      return HY_TRANSLATION_TABLE_STANDARD_NUCLEOTIDE;
    case 20:
      return HY_TRANSLATION_TABLE_STANDARD_PROTEIN;
    }
  }
  return HY_TRANSLATION_TABLE_NONSTANDARD;
}

//______________________________________________________________________________
// merge the translation tables if they are compatible, return the result,
// otherwise return nil
_TranslationTable *_TranslationTable::mergeTables(_TranslationTable *table2) {

  const unsigned char my_type = this->detectType();

  if (my_type != table2->detectType()) {
    return nil;
  }

  if (my_type == HY_TRANSLATION_TABLE_NONSTANDARD &&
      !base_set.Equal(&table2->base_set)) {
    return nil;
  }

  _TranslationTable *result = new _TranslationTable(*this);

  if (table2->tokens_added.s_length) {
    for (unsigned long i = 0; i < table2->tokens_added.s_length; i++) {
      long f = tokens_added.Find(table2->tokens_added[i]);
      if (f == HY_NOT_FOUND) {
        result->tokens_added &&table2->tokens_added[i];
        result->translations_added << table2->translations_added[i];
      } else if (translations_added(f) !=
                 table2->translations_added(i)) {
        DeleteObject(result);
        return nil;
      }
    }
    return result;
  } else {
    return result;
  }

  return nil;
}

//______________________________________________________________________________
const _String *_TranslationTable::retrieveCharacters(void) const {
  if (base_set.s_length) {
    return &base_set;
  }

  const _String *res = this->getDefaultAlphabet(this->dimension());
  return res ? res : &empty;
}

//______________________________________________________________________________
const _String *_TranslationTable::getDefaultAlphabet(const long size) {
  switch (size) {
  case 2:
    return &binary_one_char_codes;
  case 4:
    return &dna_one_char_codes;
  case 20:
    return &amino_acid_one_char_codes;
  }
  return nil;
}

//______________________________________________________________________________
bool _TranslationTable::checkValidAlphabet(const _String &try_me) {
  _String test(try_me);
  if (test.s_length > 1) {
    test.UpCase();
    _String *sorted = (_String*)test.Sort().makeDynamic();
    if (sorted && sorted->Equal(&try_me)) {
      DeleteObject(sorted);
      return true;
    }
    DeleteObject(sorted);
    warnError("_TranslationTable::checkValidAlphabet -- a sorted string with "
              "no lower case letters is required");

  } else {
    warnError("_TranslationTable::checkValidAlphabet -- at least two "
              "characters required");
  }
  return false;
}
