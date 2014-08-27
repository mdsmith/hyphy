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

#ifndef _HYTRANSLATIONTABLE_
#define _HYTRANSLATIONTABLE_
//#pragma once

#include "hy_strings.h"
#include "hy_list.h"

#define HY_TRANSLATION_TABLE_NONSTANDARD 0x000
#define HY_TRANSLATION_TABLE_STANDARD_BINARY 0x001
#define HY_TRANSLATION_TABLE_STANDARD_NUCLEOTIDE 0x002
#define HY_TRANSLATION_TABLE_STANDARD_PROTEIN 0x004
#define HY_TRANSLATION_TABLE_ANY_STANDARD                                      \
  (HY_TRANSLATION_TABLE_STANDARD_BINARY |                                      \
   HY_TRANSLATION_TABLE_STANDARD_NUCLEOTIDE |                                  \
   HY_TRANSLATION_TABLE_STANDARD_PROTEIN)

//_________________________________________________________
class _TranslationTable : public BaseObj {

public:

  /**
   * Empty constructor. Sets a few defaults
   */
  _TranslationTable(void);

  /**
   * Constructor from alphabet.
   * @param alphabet Provide the alphabet for this translation table.
   */
  _TranslationTable(const _String &);
  /* 20100618: SLKP

            - new constructor (needed to handle ExecuteCase52 /
          Simulate properly)
              which takes an alphabet string and checks to see if it's
          a standard one
              DNA/RNA/Protein or Binary

     */

  /**
   * Constructor from existing table.
   * @param t Provide a translation table to be duplicated for this
   * translation table.
   */
  _TranslationTable(_TranslationTable &);
  virtual ~_TranslationTable(void) {
    if (check_table) {
      delete [] check_table;
    }
  }

  /**
   * Make a dynamic version of this object
   */
  virtual BaseRef makeDynamic(void) const;

  /**
   * Make this translation table a duplicate of another translation table
   */
  virtual void Duplicate(BaseRefConst);

  /**
   * Return a translated version of a token
   * @param token the token provided
   * @return the code version of the token
   */
  const unsigned long tokenCode(const char) const;

  /**
   * Return a translated version of a token
   * @param token the token provided
   * @param receptacle the code recepticle to be filled.
   * @param gap_to_ones replace gap characters with all ones (ambiguity)
   * @return t/f was this token part of an added set or a custom base set?
   */
  bool tokenCode(const char, long *, const bool = true) const;

  /**
   * Return a translated version of a code
   * @param split the code provided
   * Assumes a non-unique translation of split
   * for unique - use convertCodeToLetters
   */
  char codeToLetter(long *) const;

  /**
   * Replace the current base alphabet with a given one
   * @param code the given alphabet
   */
  void addBaseSet(const _String &);

  /**
   * Split a long code into a bit string, store result in receptacle
   * @param code the code to split
   * @param receptacle for returning the split code
   */
  void splitTokenCode(long, long *) const;

  /**
   * Add a new token to the alphabet
   * @param token the token to add to the alphabet
   * @param code the code that token maps to, check to prevent base set
   * redefinition
   */
  void addTokenCode(const char, _String &);

  /**
   * Prepare a ledger for checking the translation table
   */
  void prepareForChecks(void);

  /**
   * Check for table verification ledger, return content of ledger for that
   * character
   * @param c the character to check
   * @return the value of the verification ledger, t/f.
   */
  const bool isCharLegal(const char);

  /**
   * Get character that maps to all ones, if one exists
   * @return return the all 1 character if exists, '?' otherwise
   */
  const char getSkipChar(void) const;

  /**
   * Get character that maps to all zeroes, if one exists
   * @return return the all 0 character if exists, '-' otherwise
   */
  const char getGapChar(void) const;

  /**
   * Convert a code into a string of letters
   * @param code the given code
   * @param base the length of the return string, as a define of one of the
   * basic types (nuc vs aa)
   * @return the string of letters encoded by code
   */
  _String convertCodeToLetters(long, const char);

  /**
   * Get the length of the alphabet used in translation
   * @return the length of the alphabet
   */
  const unsigned long lengthOfAlphabet(void) const;

  /**
   * Get the length of the base alphabet
   * @return the length of the base alphabet
   */
  inline const unsigned long dimension(void) const { return base_length; }

  /**
   * Get the base alphabet
   * @return the base alphabet
   */
  const _String *retrieveCharacters(void) const;

  /**
   * Get the tokens added to the alphabet
   * @return the tokens added to the alphabet
   */
  const _String &retrieveAddedTokens(void) const { return tokens_added; }

  /**
   * Reset the translation table and all objects and variables therein
   */
  void clear(void);

  /**
   * Change the base alphabet to a different data type (nucleotide vs amino
   * acid)
   * @param type a constant representing the target type
   */
  void setStandardType(const unsigned char);

  /**
   * Check that the type (nuc vs aa) of the provided pattern is the same as
   * this translation table
   * @param pattern the letters to be typed
   * @return whether or not the pattern is the same type as the translation
   * table, t/f
   */
  bool checkType(const unsigned char) const;

  /**
   * Check the type (nuc vs aa) of the translation table
   * @return the type of the translation as the value of a constant (define)
   */
  const unsigned char detectType(void) const;

  /**
   * Merge the given and this TransTable, if possible, return result
   * @param the translation table to be merged with this table
   * @return A translation table that contains the product of merging the two
   * tables
   */
  _TranslationTable *mergeTables(_TranslationTable *);

  /**
   * Get the characters in the base alphabet of length size as a string
   * @param The length of the alphabet to be matched and returned
   * @return A string containing ever character in the default alphabet of
   * size (given) size
   */
  static const _String *getDefaultAlphabet(const long);

private:

  /**
   * Check whether the given string meets the requirements of an alphabet
   * @param try_me the string to try as an alphabet
   * @return t/f does the provided alphabet meet the requirements
   */
  static bool checkValidAlphabet(const _String &);

  // number of "fundamental" tokens
  //(4 for nucl, ACGT; 20 for amino acids)
  unsigned long base_length;

  _String tokens_added, base_set;

  _hyList<long> translations_added;

  // if null - then assume default translation table;
  unsigned char *check_table;
};

extern _TranslationTable default_translation_table;
extern _String amino_acid_one_char_codes, dna_one_char_codes, rna_one_char_codes,
    binary_one_char_codes;

#endif
