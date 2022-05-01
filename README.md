Trace file format
=================
Description
In preprocessing of financial1.bz we first split our data into 23 asu number where Application specific Unit will act as a file name that we put all the related files into it. This will enable us 
After using the time window 64ms we created a new dictionary file that all the sequences that we caught with reduced vocabs.
This enabled us to find how many vocabs are in a single dictionary, we found out most them are have 5 sequences of vocab where the max is 19 sequences. This is where the limit of seq that nlp will learn. Now we build 19 LSTM cells as encoder and decoder and feed the train and test into them.
Not to mention that the input X and test all have 19 tokens this are just tokens of popular block address. There should be about 66 numbers. Training in this fashion reduces the computational load and increases our efficiency.
Overview

The trace file is composed of **variable length ACSII records, rather than
binary data**. Although this format is somewhat wasteful of storage space and
places higher CPU demands on analysis programs, it offers many advantages from a
legibility and portability standpoint.

Each record in the **trace file represents one I/O command**, and consists of
several fields. The individual fields are separated by a comma (hex 2C), with
the trace record being terminated by a newline character (\\n). There is no
special end-of-file delimiter; that function being left to the individual
operating systems.

Trace record format
-------------------

Each trace record represents a **single I/O command**. This record is divided
into five required fields, with optional fields added at the end. White space
characters, such as tabs and spaces, are optional, and may be used for visual
clarity if desired. If white space characters are present in any of the five
required fields, they may only exist between the comma character and the
beginning of the next field. White space characters may exist in any position in
the optional fields.

### Field 1: **Application specific unit (ASU) \#This might be an application number see the SPC global parameter for more info**

The ASU is a positive integer representing the application specific unit (see
the SPC Global Parameter specification document). This is a zero based,
monotonically increasing number. The first record in the trace file need not
have the ASU equal to zero, however unit zero must exist within the trace file.
If there are a total of *n* units described in the complete trace file, then the
trace file must contain at least one record for each of units *0* through *n-1*.

### Field 2: **Logical block address (LBA)**

The LBA field is a positive integer that describes the **ASU block offset** of
the data transfer for this record, where the size of a block is contained in the
description of the trace file. **This offset is zero based, and may range from 0
to n-1, where n is the capacity in blocks of the ASU**. **There is no upper
limit on this field,** other than the **restriction that sum of the address and
size fields must be less than or equal to the capacity of the ASU**.

### Field 3: Size

The size field is a positive integer that describes the **number of bytes
transferred for this record**. A value of zero is legal, the result of which is
I/O subsystem dependent. Although the majority of records are anticipated to be
**modulo 512**, this constraint is not required. There is no upper limit on this
field, other than the restriction that sum of the address and size fields must
be less than or equal to the capacity of the ASU.

### Field 4: Opcode

The opcode field is a single, case insensitive character that defines the
direction of the transfer. There are two possible values for this field:

1.  “R” (or “r”) indicates a read operation. This implies data transfer *from*
    the ASU *to* the host computer.

2.  “W” (or “w”) indicates a write operation. This implies data transfer *to*
    the ASU *from* the host computer

### Field 5: Timestamp

The timestamp field is a positive real number representing the offset in seconds
for this I/O from the start of the trace. The format of this field is “s.d”,
where “s” represents the integer portion, and “d” represents the fractional
portion of the timestamp. Both the integer and fractional parts of the field
must be present. The value of this field must be greater than or equal to all
preceding records, and less than or equal to all succeeding records. The first
record need not have a value of “0.0”.

### Field 6 (through n): Optional field(s)

Since there will undoubtedly be large amounts of additional information that is
present in the raw captured data, and since this information may well be worthy
of detailed analysis, the provision for optional fields exists in the SPC trace
files. The content of any optional field is not defined by this document, since
the meaning and content may change from one trace to another. Moreover, the
purpose of this document is to allow the creation of a standard suite of
analysis programs, and the presence (and implied absence) of optional fields
negates the concept of standardization.

Nonetheless, optional fields may be added to the each record in the trace file
by the simple expedient of inserting a field separator (comma), followed by the
field. If there is more than one field to be added, then each field must be
separated from the preceding field by the comma field separator. Optional fields
need not be present on every record of the trace file. Moreover, the number of
optional fields may change within a single trace file.

Standard analysis programs should be written so that the presence, absence, and
number of optional fields will not impact their function.

Trace file example
------------------

The following is an example of the first few record of a trace file:

>   0,20941264,8192,W,0.551706,Alpha/NT

>   0,20939840,8192,W,0.554041

>   0,20939808,8192,W,0.556202

>   1,3436288,15872,W,1.250720,0x123,5.99,test

>   1,3435888,512,W,1.609859

>   1,3435889,512,W,1.634761

>   0,7695360,4096,R,2.346628

>   1,10274472,4096,R,2.436645

>   2,30862016,4096,W,2 448003

>   2,30845544,4096,W,2.449733

>   1,10356592,4096,W,2.449733

The first record occurs 0.551706 seconds after the start of the trace file, and
is for ASU 0. The operation is to write 8,192 bytes of data to block 20,941,264.
Note the optional field that has been added to the end of the first record.

The second I/O occurs 0.002335 seconds later (0.554041 seconds into the trace),
and is an 8,192 byte write to block 20,939,840 of ASU 0.

The first I/O to ASU 1 occurs at the fourth record (1.25072 seconds into the
trace), and is a write of 15,872 bytes to block 3,436,288. Note that this record
has three optional fields, none of which is defined in this document.

Finally, note that the last two records in this example have an identical time
stamp of 2.449733. This may be due to the commands in question being issued at
exactly the same time from different host computers, or may be simply a case of
poor resolution in the data capture routines.

My Notes:

One hour = 3600 second = 450k = every 0.008 seconds on average
