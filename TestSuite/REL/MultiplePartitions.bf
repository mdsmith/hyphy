#NEXUS

BEGIN TAXA;
	DIMENSIONS NTAX = 9;
	TAXLABELS
		'RecombStrain' 'D_CD_83_ELI_ACC_K03454' 'D_CD_83_NDK_ACC_M27323' 'D_UG_94_94UG114_ACC_U88824' 'D_CD_84_84ZR085_ACC_U88822' 'B_US_83_RF_ACC_M17451' 'B_FR_83_HXB2_ACC_K03455' 'B_US_86_JRFL_ACC_U63632' 'B_US_90_WEAU160_ACC_U21135' ;
END;

BEGIN CHARACTERS;
	DIMENSIONS NCHAR = 2400;
	FORMAT
		DATATYPE = DNA
		GAP=-
		MISSING=?
	;

MATRIX
	'RecombStrain'                ACTGCAAACTTACATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGAACTTCTTGAGGAACCATAAGAGAAGATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTTTAGAGCGATTAAAAAAAAGACCCACAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCCACGTATTTAAGGAAAACATTAAGAAGAAACCAGTAAGAGGAGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCAGGCGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAAAGTTCCAAAAACGTGAAATTTATAAATCTACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAACAAATTGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCGTCAAAGTGTCTCCGGTCCAAATCGGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGACAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACGATAGAAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTAGAGGCATAGAGTCAAAAGGGGAGATGTTGCGATATGATCTGCGTGACTAAGCAATTCGGGATCGTATGAATCAAGTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTATTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATGTTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACCACCGTATGATCTACTAACAAATAATAAGAGTTAGGTAGAACGAACCCGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGGAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAACAACATCCATTTGGCATAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATTGAAATAACCGTATGGAAACCAGAAACTAAATTCAAAGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGAAGAAACCAGCGAAGATATTAGACTTAATTGATGATGAACCCCAAAATAAGAATACATCAGTAAAGGACCCTTAAACAATCATGTTAGACGCTAAATTTTAAAGGAGACAAACAATCTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAATGATACCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTGCCAGGAAACTCATGCTGAATTCAAATGTAGCAAATGAATCGAGAAAGGGTTCTCCCGGAGGAACACGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTATACCGAAAGTAATATGAGTCACTGAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATCGGGTTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGTAAGCGTGAAGTCGAAAACAACATGCTTGCGGAAGCATAAAAAGTACTTATGACACGCGTGTGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAGTGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'D_CD_83_ELI_ACC_K03454'      AATGCAAACTTACATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAAAAAAAAAAGAACTTCTTGAGGAACCATAAGAGAAGATAACCGTACAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTTTAGAGCGATTGGAAAAAAAACCCGCAACCTTGGTTTACTACGGTACATAAAAGATAACAAATTGCAAAATACGCCCACGTATTTAAGGAAAACATTAAGAAGAAATCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCATGTGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACGTGAAATTTATAAATCTACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAACAAATTGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACGAAGCAAGGGGTATGCGGCGTCAAAGTGTCTCCGGTCCAAATCGGGCAAAGATGAAGCGTAGCCCTTGGATAGAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACTATAGAAGGCTCGGTAATCAATTGAGAAAAAGGAGTAACGGTAGAGGCATAGAGTCAAAAGGGGAGATGTTGCGACATGATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTACTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTCGCGTAGCCCATTGCTTGATATATTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACCGCCGTATGATCTGCTAACAAATAATAAGAGTTAGGTAGAACGAACCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGTAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCCTAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATCGAAATAACCGTATGGAAATCAGAAACTAAATTCAATGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGTAGATACCAGCGAGGATATTAGACCTAATTGATGATGAACCCTAAAATAAGAATATATCATTAAAGGACCCTTGAACAATCATGTTAGACGCTAAATTTTAAAGGGGACAAACAATCTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAAAGATACCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTAACAGGAAACTCATGCTGAATCCAAATGTAGCAAATGAATCGAGAAAGAGTTCTCCCGGAGGAACACGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTATGCCGAAAGTAATATGAGTCACTGAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATCGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGTAAGCGTGAAGTCGAAAACAACATGCTTGCGGAAGCATAAAAAGTACTTATGACACGCGTGCGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAGTGGAGTAACTTACAATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'D_CD_83_NDK_ACC_M27323'      ACTGCAAACTTACATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGAACTTCTTGAGGAACCATAAGAGAAGATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTTTAGAGCGATTAGAAAAAAGACCCGCAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCCACGTATTTAAGGAAAACATTAAGAAGAAACCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCATGCGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACGTGAAATTTATAAATCTACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAACAAATTGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCGTCAAAGTGTCTCCGGTCCAAATCGGGCAAAGATGAAGCGTAGCCCTTGGATAGAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACGATAGAAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTAGAGGCATAGAGTCAAAAGGGGAGATGTTGCGACATGATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTACTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATATTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACCACCGTATGATCTGCTAACAAATAATAAGAGTTAGGTAGAACGAACCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGACACAAGTAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAACTCTCCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCATAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATCGAAATAACCGTATGGAAATCAGAAACTAAATTCAATGATAAACATGTCTCGGGAAAAGTGTAGTCAAAAAAGTAGATACCAGCGAGGATATTAGACCTAATTGATGATGAACCCTAAAATAAGAATACATCATTAAAGGACCCTTGAACAATCATGTTAGACACTAAATTTTAAAGGAGACAAACAATCTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAAAGATACCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTACCAGGAAACTCATGCTGAATCCAAATGTAGCAAATGAATCGAGAAAGAGTTCTCCCGGAGGAACACGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTATACCGAAAGTAATATGAGTCACTGAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATCGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGTAAGCGTGAAGTCGAAAACAACATGCTTGCGGAAGCATAAAAAGTACTTATGACACACGTGTGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAGTGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'D_UG_94_94UG114_ACC_U88824'  ACTGCAAACTTACATGAAGAATGAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAAGTTATACCTCGAGAAAAAAAAGTACTTCTTGAGGAACCATAAGAGAAAATAATCGTGCAAGTATGATATGAACAAAAATTTGAATTGAAAGGACTGTATAGAGCGATTAAAAAAAAGACCCGCAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCCACGTATTTAAGGAAAACATTAAGAAGAAACCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACACAGGCAGGCGTCCAAGTTAAGACCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATCAAAAGATACAAAGATGAGCACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGGAGGAATCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACATGAAATCTATAAATATACGATAATGAGGGACCAACGGAAGAAAACGATATGGGAAGCGAAAGTAAATAGTTTTCGTAAACACAGACAAGTGACATCACCTCAAAATACGAGTTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCATCAAAGTGTCTCTGGTCCAAATCCGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTATTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTACCAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCGGTAGAAAATACGATAGAAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTGAAGGCATAGAGTCAAAAAGGGAGATGTTGCGATATGATCTGCGTGACTAAGCAATTCGGGATCGTATGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTATTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATGTTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACCACCGTATGATCTACTAACAAATAATAAGAGTTAAGTAGAACGAACCCGAAACTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGGAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAACAACATCCATTTGACATAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAACGTGATGCGGATTGAAATAACCGTATGGAAACCAGATAATAATTTCAAAGATAAACATGTCTCGGGAAAAGTGTAATAAAAAAAGAAGAAACCAGCGAAGATATTAGACTTAATTGATGATGGACCCCAAAATAAGAATACATAATTAAAGGACCCTTAAACAATCATGTTAGACGCTAAATTTTAAAGGAGACAAACAATCTTGATACGAGGCTCACAAGACAGAATGAAAAACCGTGTTTCATAATAGAAAAGCACAATGATGATATCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTGCCAGGAAACTCATGCTGAATTCAAATGTAGCAAATGAATCGAGAAAGAGTTCTCCTGGAGGAACACGCATAGGCAAAATAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTGTACCGAAAGTAATATGAGTCACTGAAAGATAAATAAAAATAAACTTTCAGCAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATAGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACATCCAAAAAATCTGAATTTACGCCGCATGAGTAAGCGTGAAGTCGAAGACAACGTGCTTGCGGAAGCATAAAAAGTACTTATGACACGCGTGTGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCTAGAGGAGTGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'D_CD_84_84ZR085_ACC_U88822'  ACTGCAAACTTACATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGAACTTCTTGAGGAACCATAAGAGAAAATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTATAGAGCGATTAAAAAAAAGACCCGCAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCAACGTATTTAAGGAAAACATTAAGAAGAAATCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCAGGCGTCCAAGTTAAGGCCGGGTAATTTCCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCGAATGGAAGGTAAATAACGATGCCAGGTCCAGGCAGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACATGAAATTTATAAATATACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAATAAATAGTTTCCGTAAACACAGACGAGTGACATCACCTCAAAAAACGAGTTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCATCAAAGTGTCTCTGGTCCAAATCCGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCATAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACGATAGGAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTGAAGGCATAGAGTCAAAAAGGGAGATATTGCGACATGATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGATCCAGACCTAAACTAGAAATTGAGAAACTATTCTGCAATCGAGGTCAATCTAGCAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATATTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACCACCGTATGATCTACTAACAAATAATAAGAGTTAGGTAGAACGAACCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGTAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCATAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATTGAAATAACCGTATGGAAACCAGAAACTAAATTCAAAGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGAAGAAACCAGCGAAGATATTAGACTTAATTGATGATGAACCCTAAAATAAGAATACATCATTAAAGGACCCTTAAACAATCATGTTAGACGCTATATTTTAAAGGAGACAAACAATCTTAATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAAAGATACCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTACCAGGAAACTCATGCTGAATCCAAATGTAGCAAATGAATCGAGAAAGAGTTCTCCCGGAGGAACACGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGTAAAAAGATCCTTGGCCGAACCTATACCGAAAGTAATATGAGTCACTGAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATCGGATTGACAATAATGACACAACGGTCTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAAAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGTAAGCGTGAAGTCGAAAACAACGTGCTTGCGGAAGCATAAAAAGTACTTATGGCACGCGTGAGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAGTGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'B_US_83_RF_ACC_M17451'       ACTGCAAACTTACATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGACCTTCTTGAGGAACCATAAGATAAAATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGCATAGAGCGATTAAAAAAAAGACCCGCACCCTTGGTTTACTACGGTACCTAAAAGCTAACAAATTGCAAAATACGCCCACGTATTTAAGGAAAATATTAGGAAGAAACCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGAATCAAAGATCTACGGTGTAACACATTATACATAGGCAGGCGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAATGATGCCAGGTCCAGGCTGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACGTGAAATTTATAAATATACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAATAAATAGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCATCAAAGTGTCTCTGGTCCAAATCGGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAAACATAAAGATGACGAACATGCTGGGTAAATCCGTAGAAAATACGATAGAAGGCTCGGTGATCCATTGAGAAAAAGGAGTAACGGTGAAGGCATAGAGTCAAAAAGGGAGATGTTGCGACATGATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTACTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATATTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACTACCGTATGATCTGCTAACAAATAATAAGAGTTAAGTAGAACGAACCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGTAAGCAAATAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTGCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCATAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATTGAAATAACCGTATGGAAACCAGAAACTAAATTCAATGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGTAGAAACCAGCGAAGATATTAGACTTAATTGATGATGAACCCTAAAATAAGAATACATCATTAAAGGACCCTTAAACAATCATGTTAGACGCTAAATTTTAAAGGAGACAAACAATCTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAACAGAAAAGCACTATAAAGATAACCAGACGGTAAATTACGGGAAGGCTGTAAGGATGAATGCTACCAGGAAACTCATGCTAAATCCAAATGTAGCAAATGAATCGAGAAAGAGTTCTCCCGGAGGAACGCGCATAGGCAAATTAAATGATGAGAGCTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTATACCGAAAGTAATATGAGTTACTGAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAATACGGTGATCGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGCAAGCGTGAAGTCGAAAACAACGTGCTGGCGGAAGCATAAAAAGTACTTATGGCACGCGTATGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAAGAATGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'B_FR_83_HXB2_ACC_K03455'     ACTGCAAACTTACATGAAAAATAAAACGGAGCGATTGGTAGAGAAATTAGAAGACTACATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGACCTTCTTGAGGAACCATAAGAGAAAATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTATAGAGCGATTAAAAAAAACACCCGCAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCCACCTATTTAAGGAAAACATTAGGAAGAAACCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCAGGCGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACGTGAAATTTATAAATATACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAATAAATAGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCATCAAAGTGTCTCTGGTCCACATCGGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACGATAGAAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTGGAGGCATAGAGTCAAAAAGGGAGATGTTGCGACATGATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTACTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATATTCCTAAGTAACTAGCAATAAATCCAACGTTAGGTCACCACCACCGTATGATCTGCTAACAAATAATAAGAGTTAGGTAGATCGAAGCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGTAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCATAGAGATTATAGTTTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACAAATGTGATACGGATTGAAATAACCGTATGGAAACCAGAAACTAAATTCAATGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGTAGAAACCAGCGAAGATATTAGACTTAATTGATGATGAACCCTAAAATAAGAATACATCATTAAAGGACCCTTGAACAATCATGTTAGACGCTAAATTTTAAAGGAGACAAACAATTTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAAAGATACCCAGACGGCAAATTCCGGGAAGGCTGTAAGGATGAATGCTACCAGGAAACTCATGCTGAATCCAAATGTAGCAAATGAATCGAGAAAGAGTTCTCCCGGAGGAACGCGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTATACCGAAAGTAATATGAGTTACTCAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATCGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGCAAGCGTGAAGTCGAAAACAACGTGCTGGCGGAAGCATAAAAAGTACTTATGGCACGCGTGTGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAATGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'B_US_86_JRFL_ACC_U63632'     ACTGCAAACTTACATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGACCTTCTTGAGGAACCATAAGAGAAAATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTATAGAGCGATTAAAAAAAAGACCCGCAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCCACCTATTTAAGGAAAACATTAGGAAGAAACCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCAGGCGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATAAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAGTGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGGAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACGTGAAATTTATAAATCTACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAACAAATAGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACTAAGCAAGGGGTATGCGGCGTCAAAGTGTCTCCGGTCCAAATCGGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATAAGATCAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTCGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACGATAGAAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTGGAGGCATAGAGTCAAAAGGGGAGATGTTGCGACATGATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAGAAACTACTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAGCCCATTGCTTGATATATTCCTAAGTAACTAGCAACAAATCCAACGTTAGGTCACCACCACCGTATGATCTGCTAACAAATAATAAGGGTTAGGTAGAACGAACCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTAGAGACAAGTAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCATAGAGATTATAGATTGTAGGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATCGAAATAACCGTATGGAAATCAGAAACTAAATTCAATGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGTAGATACCAGCGAGGATATTAGACTTAATTGATGATGAACCCTAAAATAAGAATACATCATTAAAGGACCCTTGAACAATCATGTTAGACGCTAAATTTTAAAGGAGACAAACAATCTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAAAGATACCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTACCAGGAAACTCATGCTGAATCCAAATGTAGCATATGAATCGAGAAAGAGTTCTCCCGGAGGAACGCGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACGTATACCGAAAGTAATATGAGTTACTCAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGATGATCGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGCAAGCGTGAAGTCGAAAACAACGTGCTGGCGGAAGCATAAAAAGTACTTATGGCACGCGTGTGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAATGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA
	'B_US_90_WEAU160_ACC_U21135'  ACTGCAAACTTATATGAAAAATAAAACGGCGCGATTGGTAGAGAAATTAGAAGACTATATCATTGGACTAAAAAATGAAAAATAGACACTGCACCTCAGTAAAGCAGAGCAGTGACGGTATGTAAAGATCAAAGAAGGGTGTAGGTTATACCTCGAGAAAAAAAAGACCTTCTTGAGGAACCATAAGAGAAAATAACCGTGCAAGTATGATATAACCAAAAATTTGAATTGGAAGGACTGTATAGAGCGATTAAAAAAAAGACCCGCAACCTTGGTTTACTACGGTACATAAAAGCTAACAAATTGCAAAATACGCCCACGTATTTAAGGAAAACATTAGGAAGAAACCAGTAAGAGGGGCTTAGTAAAATCTATATAGAAAAGAAGCATCAAAGATCTACGGTGTAACACATTATACATAGGCAGGCGTCCAAGTTAAGGCCGGGTAATTACCAAAGTGAACGTCTACAGCAAAGACAATAGAATAAAAAGATTAAAAAATGAGAACCGGGGAGTGGGAGAGTCTAATGGAAGGTAAATAACGATGCCAGGTCCAGGCTGGTAGCAGGAGTCGCTAGAACTCATCGAATTTCCTTGAAATAGCAACCAGACCCAGAGTTCCAAAAACGTGAAATTTATAAATATACGATAATGAGGGACTAACGGAAGAAAACGATATGGGAAGCGAAAATAAATAGTTTCTGTAAACACAGACAAGTGACATCACCTCGAAATACGAATTAGCAAGTAAGGGTATACTGAGCAAGGGGTATGCGGCATCAAAGTGTCTCTGGTCCAAATCGGGCAAAGATGAAGCGTAGCCCTTGGATAAAACTATAAGGCTGGCTAAGAAGACCAAAACATTAGTTTATAGTGGGGAAAGTCAAAGAAAGTCTAAACACGATTAGTGTTCGTATGAGATGAGAACGAAATCCAAATAATCCGCCAAATGTGTGCGGTGGCTAACAAAATACGGTACGACATTCGAACAAAGACATAAAGATGACAAACATGCTGGGTAAATCCGTAGAAAATACGATAGAAGGCTCGGTGATCAATTGAGAAAAAGGAGTAACGGTGGAGGCATAGAGTCAAAAAGGGAGATGTTGCGACATAATCTGCGTGACTAAGCAATTCGGGATCGTACGAATCAACTGCAAGGTCCAGACCTAAACTAGAAATTGAAAAACTACTCTGCAATCGAGGTCAATCTAACAGGCAGGAGATTTAGCGTAACCCATTGCTTGATATATTCCTAAGTAATTAGCAACAAATCCAACGTTAGGTCACCACCACCGTATGATCTGCTAACAAATAATAAGAGTTAGGTAGAACGAACCGGAAGCTGTTGTTATTAAAAGGATGCTAATATGGAACGGTGGAGACAAGTAAGCAAGTAGCATTCCTTACAGATACCTTTAATTGTGAGCTCTCCACTAAAACTATGGTACGGGGTAGCAACATCCATTTGGCATAGAGATTATAGTTTGTAAGAAAGGTGAGGATAGATGCTGTGGGATAAGAGCTGGAATCTCAAACAGATTACGAATGTGATGCGGATTGAAATAACCGTATGGAAACCAGAAACTAAATTCAATGATAAACATGTCTCGGGAAAAGTGTAGTAAAAAAAGTAGAAACCAGCGAAGATATTAGACTTAATTGATGATGAACCCTAAAATAAGAATACATCATTAAAGGACCCTTGAACAATCATGTTAGACGCTAAATTTTAAAGGAGACAAACAATCTTGATACGAGGCTCACAAGACGGAATGAAAAACCGTGTTTCATAATAGAAAAGCACTATAAAGATACCCAGACGGTAAATTCCGGGAAGGCTGTAAGGATGAATGCTACCAGGAAACTCATGCTGAATCCAATTGTAGCAAATGAATCGAGAAAGAGTTCTCCCGGAGGAACGCGCATAGGCAAATTAAATGATGAGAACTAAGGTAGCAAAAAAAGAGAAAAAAGATCCTTGGCCGAACCTATACCGAAAGTAATATGAGTTACTCAAAGGTAAATAAAAATAAACTTTCAACAGGGAAGTAATCACAGACAAGTTACTCACAAATACATTAACACGGTGATCGGATTGACAATAATGACACAACGGTTTGACCTGGTATGTCCACTGATATTAGGTAACAAAACAAAATAATAGAACTTTACGTGATAGAAAGAACTTCCAAAAAATCTGAATTTACGCCGCATGAGCAAGCGTGAAGTCGAAAACAACGTGCTGGCGGAAGCATAAAAAGTACTTATGGCACGCGTGTGAGTTGGTACGTAAGAGCTAAATAGTATGATAGACCAGGTGCCTCCAGAGGAATGGAGTAACTTACGATGTGGTAGCGTAAGAGCAGTTAAAAGAATCAGATTAATAAGAACCCCGAA;
END;

BEGIN HYPHY;


global CT=1.22367;
global AT=0.284002;
global CG=0.390749;
global AC=0.369337;
global GT=0.142059;
global betaP=85;
betaP:>0.05;
betaP:<85;
global betaQ=4.32871;
betaQ:>0.05;
betaQ:<85;
global alpha=0.197532;
alpha:>0.01;
alpha:<100;
pc.weights={
{    0.333333333333}
{    0.333333333333}
{    0.333333333333}
}
;


category pc=(3,pc.weights,MEAN,_x_^(betaP-1)*(1-_x_)^(betaQ-1)/Beta(betaP,betaQ),IBeta(_x_,betaP,betaQ),0,1,IBeta(_x_,betaP+1,betaQ)*betaP/(betaP+betaQ));
category c=(4,pc,MEAN,GammaDist(_x_,alpha,alpha),CGammaDist(_x_,alpha,alpha),0,1e+25,CGammaDist(_x_,alpha+1,alpha));

GTR_Matrix={4,4};
GTR_Matrix[0][1]:=AC*t*c;
GTR_Matrix[0][2]:=t*c;
GTR_Matrix[0][3]:=AT*t*c;
GTR_Matrix[1][0]:=AC*t*c;
GTR_Matrix[1][2]:=CG*t*c;
GTR_Matrix[1][3]:=CT*t*c;
GTR_Matrix[2][0]:=t*c;
GTR_Matrix[2][1]:=CG*t*c;
GTR_Matrix[2][3]:=GT*t*c;
GTR_Matrix[3][0]:=AT*t*c;
GTR_Matrix[3][1]:=CT*t*c;
GTR_Matrix[3][2]:=GT*t*c;

baseFreqs={
{    0.399398148148}
{    0.163842592593}
{    0.223981481481}
{    0.212777777778}
}
;
Model GTR_Model=(GTR_Matrix,baseFreqs);

UseModel (GTR_Model);
Tree tree_0=((((RecombStrain,(D_CD_83_ELI_ACC_K03454,D_CD_83_NDK_ACC_M27323)Node5)Node3,D_CD_84_84ZR085_ACC_U88822)Node2,D_UG_94_94UG114_ACC_U88824)Node1,(B_US_83_RF_ACC_M17451,B_US_90_WEAU160_ACC_U21135)Node10,(B_FR_83_HXB2_ACC_K03455,B_US_86_JRFL_ACC_U63632)Node13);

UseModel (GTR_Model);
Tree tree_1=((((RecombStrain,(D_CD_83_ELI_ACC_K03454,D_CD_83_NDK_ACC_M27323)Node5)Node3,B_US_86_JRFL_ACC_U63632)Node2,B_US_90_WEAU160_ACC_U21135)Node1,((D_UG_94_94UG114_ACC_U88824,D_CD_84_84ZR085_ACC_U88822)Node11,B_US_83_RF_ACC_M17451)Node10,B_FR_83_HXB2_ACC_K03455);

UseModel (GTR_Model);
Tree tree_2=(((((RecombStrain,D_UG_94_94UG114_ACC_U88824)Node4,D_CD_84_84ZR085_ACC_U88822)Node3,B_US_83_RF_ACC_M17451)Node2,B_FR_83_HXB2_ACC_K03455)Node1,((D_CD_83_ELI_ACC_K03454,D_CD_83_NDK_ACC_M27323)Node11,B_US_86_JRFL_ACC_U63632)Node10,B_US_90_WEAU160_ACC_U21135);

UseModel (GTR_Model);
Tree tree_3=(RecombStrain,(D_CD_83_ELI_ACC_K03454,(D_UG_94_94UG114_ACC_U88824,(D_CD_84_84ZR085_ACC_U88822,(B_US_83_RF_ACC_M17451,((B_FR_83_HXB2_ACC_K03455,B_US_86_JRFL_ACC_U63632)Node11,B_US_90_WEAU160_ACC_U21135)Node10)Node8)Node6)Node4)Node2,D_CD_83_NDK_ACC_M27323);

tree_2.B_US_83_RF_ACC_M17451.t=0.02755683682993398;
tree_2.Node1.t=0;
tree_2.B_US_86_JRFL_ACC_U63632.t=0.01326059287271811;
tree_2.Node10.t=0.01326143467048369;
tree_2.B_US_90_WEAU160_ACC_U21135.t=0.01987764324853837;
tree_2.Node11.t=0.003026459698870693;
tree_2.B_FR_83_HXB2_ACC_K03455.t=0.02329267821605396;
tree_2.Node2.t=0.002800333307564091;
tree_2.Node4.t=0.03500361914558993;
tree_2.D_UG_94_94UG114_ACC_U88824.t=0.04889981365158647;
tree_2.RecombStrain.t=0.006715808089228459;
tree_2.D_CD_84_84ZR085_ACC_U88822.t=0.01332295638171858;
tree_2.D_CD_83_NDK_ACC_M27323.t=0.01328712408506268;
tree_2.D_CD_83_ELI_ACC_K03454.t=0.01996826383490629;
tree_2.Node3.t=0.01383818615750967;
tree_3.RecombStrain.t=0.009992697906091448;
tree_3.B_US_86_JRFL_ACC_U63632.t=0.0099749493969674;
tree_3.Node2.t=0;
tree_3.Node8.t=0.02636648254239594;
tree_3.Node6.t=0.004563116148969946;
tree_3.B_US_90_WEAU160_ACC_U21135.t=0;
tree_3.B_FR_83_HXB2_ACC_K03455.t=0;
tree_3.B_US_83_RF_ACC_M17451.t=0.02027530226140629;
tree_3.D_CD_83_NDK_ACC_M27323.t=0.004993954698447391;
tree_3.Node4.t=0.005135340264405711;
tree_3.D_UG_94_94UG114_ACC_U88824.t=0.04793012238098922;
tree_3.Node11.t=0;
tree_3.D_CD_84_84ZR085_ACC_U88822.t=0.02087877906855073;
tree_3.D_CD_83_ELI_ACC_K03454.t=0.01507507532185939;
tree_3.Node10.t=0.005001849275258297;
tree_1.B_US_90_WEAU160_ACC_U21135.t=0.02321828136492775;
tree_0.Node10.t=0;
tree_0.Node2.t=0;
tree_0.Node5.t=0.01210808669935977;
tree_0.B_US_86_JRFL_ACC_U63632.t=0;
tree_0.B_FR_83_HXB2_ACC_K03455.t=0.01816066369929858;
tree_0.B_US_83_RF_ACC_M17451.t=0.03742153847361689;
tree_0.Node1.t=0.01195896810357335;
tree_0.D_CD_83_NDK_ACC_M27323.t=0;
tree_1.Node10.t=0.003353456605947262;
tree_0.D_UG_94_94UG114_ACC_U88824.t=0.05673213356293653;
tree_0.RecombStrain.t=0.01200894050847036;
tree_0.D_CD_84_84ZR085_ACC_U88822.t=0.01206640617488343;
tree_0.D_CD_83_ELI_ACC_K03454.t=0.05053803232401123;
tree_0.Node3.t=0.01211693401151765;
tree_0.B_US_90_WEAU160_ACC_U21135.t=0.005897424801452161;
tree_1.Node11.t=0.01943622215056805;
tree_1.Node2.t=0.01930639918536271;
tree_1.Node5.t=0.00367221191230085;
tree_1.B_US_86_JRFL_ACC_U63632.t=0.003556907631311533;
tree_1.B_FR_83_HXB2_ACC_K03455.t=0.003761914794499745;
tree_1.B_US_83_RF_ACC_M17451.t=0.01579657810015503;
tree_1.Node1.t=0;
tree_1.D_CD_83_NDK_ACC_M27323.t=0;
tree_1.D_UG_94_94UG114_ACC_U88824.t=0.05405801345442108;
tree_1.RecombStrain.t=0.007591494164654542;
tree_0.Node13.t=0.005916039435863761;
tree_1.D_CD_84_84ZR085_ACC_U88822.t=0.03313660975482054;
tree_1.D_CD_83_ELI_ACC_K03454.t=0.01143840296527942;
tree_1.Node3.t=0.007767398822293749;
DataSet ds = ReadDataFile(USE_NEXUS_FILE_DATA);
DataSetFilter part_0 = CreateFilter(ds,1,"0-427","0-2,4,3,5,8,6,7");
DataSetFilter part_1 = CreateFilter(ds,1,"428-1105","0-2,7,8,3-6");
DataSetFilter part_2 = CreateFilter(ds,1,"1106-1891","0,3-6,1,2,7,8");
DataSetFilter part_3 = CreateFilter(ds,1,"1892-2399","0,1,3-8,2");
LikelihoodFunction multiPart = (part_0,tree_0,part_1,tree_1,part_2,tree_2,part_3,tree_3);

Optimize(res_multiPart,multiPart);

ExecuteAFile ("../Shared/REL_utils.bf");
marginalErrors = checkMarginalReconstruction ( expectedConditionalProbabilities, exepectedRateClassAssignments, "flu_LF");

/* test epilogue */
	timeMatrix = endTestTimer 				  (_testDescription);
	if (logTestResult (Abs (res_flu_LF[1][0] - _expectedLL) < 0.01 && marginalErrors[0] < 0.0001 && marginalErrors[1] == 0))
	{
		return timeMatrix;
	}
	return 0;
/* end test epilogue */


END;
