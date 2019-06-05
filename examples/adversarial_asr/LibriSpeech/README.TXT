1. General information
======================

LibriSpeech is a corpus of read speech, based on LibriVox's public domain
audio books. Its purpose is to enable the training and testing of automatic
speech recognition(ASR) systems. 


2. Structure
============

The corpus is split into several parts to enable users to selectively download
subsets of it, according to their needs. The subsets with "clean" in their name
are supposedly "cleaner"(at least on average), than the rest of the audio and
US English accented. That classification was obtained using very crude automated 
means, and should not be considered completely reliable. The subsets are
disjoint, i.e. the audio of each speaker is assigned to exactly one subset.

The parts of the corpus are as follows:

* dev-clean, test-clean - development and test set containing "clean" speech.
* train-clean-100 - training set, of approximately 100 hours of "clean" speech
* train-clean-360 - training set, of approximately 360 hours of "clean" speech
* dev-other, test-other - development and test set, with speech which was 
                          automatically selected to be more "challenging" to
                          recognize
* train-other-500 - training set of approximately 500 hours containing speech
                    that was not classified as "clean", for some (possibly wrong)
                    reason
* intro - subset containing only the LibriVox's intro disclaimers for some of the
          readers.
* mp3 - the original MP3-encoded audio on which the corpus is based
* texts - the original Project Gutenberg texts on which the reference transcripts
          for the utterances in the corpus are based.
* raw_metadata - SQLite databases which record various pieces of information about 
                 the source text/audio materials used, and the alignment process.
                 (mostly for completeness - probably not very interesting or useful)

2.1 Organization of the training and test subsets
-------------------------------------------------

When extracted, each of the {dev,test,train} sets re-creates LibriSpeech's root
directory, containing some metadata, and a dedicated subdirectory for the subset
itself. The audio for each individual speaker is stored under a dedicated 
subdirectory in the subset's directory, and each audio chapter read by this
speaker is stored in separate subsubdirectory. The following ASCII diagram 
depicts the directory structure:


<corpus root>
    |
    .- README.TXT
    |
    .- READERS.TXT
    |
    .- CHAPTERS.TXT
    |
    .- BOOKS.TXT
    |
    .- train-clean-100/
                   |
                   .- 19/
                       |
                       .- 198/
                       |    |
                       |    .- 19-198.trans.txt
                       |    |    
                       |    .- 19-198-0001.flac
                       |    |
                       |    .- 14-208-0002.flac
                       |    |
                       |    ...
                       |
                       .- 227/
                            | ...



, where 19 is the ID of the reader, and 198 and 227 are the IDs of the chapters
read by this speaker. The *.trans.txt files contain the transcripts for each
of the utterances, derived from the respective chapter and the FLAC files contain
the audio itself.

The main metainfo about the speech is listed in the READERS and the CHAPTERS:

- READERS.TXT contains information about speaker's gender and total amount of
  audio in the corpus.

- CHAPTERS.TXT has information about the per-chapter audio durations.

The file BOOKS.TXT makes contains the title for each book, whose text is used in
the corpus, and its Project Gutenberg ID.

2.2 Organization of the "intro-disclaimers" subset
--------------------------------------------------

This part of the data contains simply the LibriVox's intro disclaimers that were
successfully extracted, using a slight modification of the alignment algorithms
used to derive the test training sets. The standard LibriVox disclaimer is:

"This is a LibriVox recording. All LibriVox recordings are in the public domain. 
 For more information, or to volunteer, please visit: librivox DOT org"

As is the case for the training and test sets, there is one subdirectory for
each reader, and a subsubdirectory for each of the chapters, read by this speaker
for which the announcement was successfully extracted.


2.3 Organization of the "original-mp3" subset
---------------------------------------------

This part contains the original MP3-compressed recordings as downloaded from the
Internet Archive. It is intended to serve as a secure reference "snapshot" for 
the original audio chapters, but also to preserve (most of) the information both
about audio, selected for the corpus, and audio that was discarded. I decided to
try make the corpus relatively balanced in terms of per-speaker durations, so
part of the audio available for some of the speakers was discarded. Also for the
speakers in the training sets, only up to 10 minutes of audio is used, to 
introduce more speaker diversity during evaluation time. There should be enough 
information in the "mp3" subset to enable the re-cutting of an extended 
"LibriSpeech+" corpus, containing around 150 extra hours of speech, if needed.

The directory hierarchy follows the already familiar pattern. In each
speaker directory there is a file named "utterance_map" which list for each
of the utterances in the corpus, the original "raw" aligned utterance.
In the "header" of that file there are also 2 lines, that show if the
sentence-aware segmentation was used in the LibriSpeech corpus(i.e. if the
reader is assigned to a test set) and the maximum allowed duration for
the set to which this speaker was assigned.

Then in the chapter directory, besides the original audio chapter .mp3 file,
there are two sets of ".seg.txt" and ".trans.txt" files. The former contain
the time range(in seconds) for each of the original(that I called "raw" above)
utterances. The latter contains the respective transcriptions. There are two
sets for the two possible segmentations of each chapter. The ".sents"
segmentation is "sentence-aware", that is, we only split on silence intervals 
coinciding with (automatically obtained) sentence boundaries in the text.
The other segmentation was derived by allowing splitting on every silence
interval longer than 300ms, which leads to better utilization of the aligned
audio.

2.4 Organization of the "text" subset
-------------------------------------

This part just contains one subdirectory, with name equal to the ID of the
text in Project Gutenberg's database, for each book. The books are also
separated in directories by their encoding-- could be either ASCII or UTF-8.
The sole purpose of this subset is to be a permanent snapshot of the original
text used for LibriSpeech's construction.


2.5 Organization of the "raw-metadata" part
-------------------------------------------

Contains just few SQLite databases. Some of the more important bits of
information from this tables are described in the README file within
the "raw_data" subdirectory.


Acknowledgments
===============

First and foremost, I would like to thank the thousands of Project Gutenberg
and LibriVox volunteers, without whose contributions the LibriSpeech corpus 
would not have existed.
The successful completion of this project would have been much more difficult,
and the quality of the finished corpus much worse, if it wasn't for the
generous support and the many helpful advice, provided by Daniel Povey - thanks, Dan!
I would also like to express my gratitude to Tony Robinson, for the very
interesting, and useful discussions on the long audio alignment problem, that
we had some time ago.
Thanks also to Guoguo Chen and Sanjeev Khudanpur, with whom we are collaborating
on a (yet-to-be-published) paper on the corpus, and who helped to improve
the LibriSpeech's example scripts in Kaldi.

---
Vassil Panayotov,
Oct. 2, 2014 
