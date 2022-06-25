# NEMESYS and NEMETYL

#### NEMESYS: NEtwork MEssage Syntax analysYS
and 
#### Format Match Score

**Paper authors:** Stephan Kleber, Henning Kopp, and Frank Kargl, *Institute of Distributed Systems, Ulm University*

https://www.usenix.org/conference/woot18/presentation/kleber


###### On usage please cite as:

*S. Kleber, H. Kopp, and F. Kargl*: „NEMESYS: Network Message Syntax Reverse Engineering by Analysis of the Intrinsic Structure of Individual Messages“, in 12th USENIX Workshop on Offensive Technologies, Baltimore, MD, USA, 2018.

[BibTeX](https://www.usenix.org/biblio/export/bibtex/220576)



#### NEMETYL: NEtwork MEssage TYpe identification by aLignment

**Paper authors:** Stephan Kleber, Rens W. van der Heijden, and Frank Kargl, *Institute of Distributed Systems, Ulm University*

https://arxiv.org/abs/2002.03391

###### On usage please cite as:

*S. Kleber, R. W. van der Heijden, and F. Kargl:* „Message Type Identification of Binary Network Protocols using Continuous Segment Similarity“, in IEEE International Conference on Computer Communications, 2020.






## Release: INFOCOM2020

**Code author:** Stephan Kleber ([stephan.kleber@uni-ulm.de](mailto:stephan.kleber@uni-ulm.de)), *Institute of Distributed Systems, Ulm University*

**NEMESYS** is a novel method to infer structure from network messages of binary protocols. The method derives field boundaries from the distribution of value changes throughout individual messages. Our method exploits the intrinsic features of structure which are contained within each single message
instead of comparing multiple messages with each other. 

Additionally, this repository contains the reference implementation for calculating the **Format Match Score**: It is the first quantitative measure of the quality of a message format inference.

**NEMETYL** is a novel method for discriminating protocol message types from each other. It uses structural features of binary protocols inferred by NEMESYS to accurately recognize structural patterns and cluster messages based on their common structure.

NEMESYS, FMS, and NEMETYL are indented to be used as a library to be integrated into your own scripts.
However, you can also use it interactively with your favorite python shell.
Have a look into `nemesys.py`, `nemesys_fms.py`, resp. `nemetyl_align-segments.py` to get an impression of the basic functionality and how to call it.



## Disclaimer

This is highly experimental software and by no means guaranteed to be fit for productive use. In particular, it has not been tested with varying platforms, python environments, and dependencies. If you run into any problems during deployment and usage, please feel highly encouraged to provide feedback or ask questions via eMail ([stephan.kleber@uni-ulm.de](mailto:stephan.kleber@uni-ulm.de)), via github issue, or a pull request.




## Requirements
* Python 3
* libpcap for pcapy: `apt-get install libpcap-dev libpq-dev`
* Install packages listed in requirements.txt: `pip install -r requirements.txt`
	* This necessitates to install libpcap for pcapy: `sudo apt-get install libpcap-dev`
* Manual install of Netzob from the ["fix-layer-build" branch](git@github.com:skleber/netzob.git)
  -- ~~currently NOT the official~~ [~~"next" branch~~](https://github.com/netzob/netzob/tree/next/netzob)! --
  (the current Netzob version available in the official repository and via PyPI lacks some required fixes): 
    * clone Netzob next branch to a local folder: `git clone --single-branch -b next https://github.com/netzob/netzob.git` 
    * install it: `python setup.py install`
* [tshark](https://www.wireshark.org/docs/man-pages/tshark.html) version 2.x or 3.x (tested with: 2.2.6, 2.6.3, 2.6.5, 2.6.8, 3.2.3, 3.2.5)
  (possibly other versions, depending on the compatibility of the JSON-output format of dissected messages,
  please report further working versions e. g., per github issue)  
  *Note: NEMESYS can be used without tshark as long as FMS validation (in package `validation`) 
  against a real dissector is NOT required.*
  
  Place your user in the "wireshark" group to enable tshark to run: `sudo gpasswd -a $USER wireshark`


## Dockerfile

There is a Dockerfile provided so that you can use Nemere without worrying about dependencies. The following code builds the image and starts a container with the directory bind as a volume so that input files can be easily used and generated reports can be easily accessed even after stopping the container.

```
git clone https://github.com/vs-uulm/nemesys.git
cd nemesys
docker build . -t nemere:latest
docker run -ti --mount type=bind,source=$(pwd),target=/nemere/ nemere:latest
```

For the use of FMS where tshark is going to be used, we need to add the `--privileged` flag to the run command above.


## Sample scripts
There are several scripts to provide a starting point to working with NEMESYS and FMS.
The short description in this section provides an overview of their functions.


### Globally available options
All scripts provide these command line options:

* Get help for the command line options by calling each script only with parameter `-h`
* You select the network layer to analyze by `-l [#]` 
  and optionally `-r` making the given layernumber relative to the IP layer (if any in the trace).
  **It is highly recommended to always include these options since they are a common cause for usage errors.**
  To select the application layer above any transport protocol (regardless whether TCP, UDP, 
  or any other protocol known to tshark) use `-r -l2`



### prep_*
PCAP preparation scripts:

* `prep_deduplicate-trace.py pcapfilename`
  Detect identical payloads and de-duplicate traces, ignoring encapsulation metadata.


### check_*
Basic checks whether PCAPs are parseable:

*relevant only for validation of NEMESYS output by FMS, not NEMESYS itself*

The tshark-dissected fields that are contained in the PCAPs need to be known to the message parser.
Therefore, validation.messageParser.ParsingConstants needs to be made aware of any field occuring in the traces.

* `check_parse-pcap.py pcapfilename`
  Parse a PCAP file and print its dissection for testing. This helps verifying if there are any unknown fields 
  that need to be added to validation.messageParser.ParsingConstants.
  Before starting to validate/use FMS with a new PCAP, first run this check and solve any errors.
  
  

### nemesys_*
Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write FMS and other evaluation data to reports and plots.

* `nemesys.py pcapfilename`
  Run NEMESYS and open an interactive python shell (IPython) that provides access to the analysis results.

* `nemesys_fms.py pcapfilename`
  Run NEMESYS and validate it by calculating the FMS for each inferred message.
  Writes the results as text files to a timestamped subfolder of `reports/` 
  
* `nemesys_field-deviation-plot.py pcapfilename`
  Run NEMESYS and validate it by visualizing the deviation of the true format from each inferred message.
  Writes the results as plots in PDF files to a timestamped subfolder of `reports/` 



### nemetyl_*

Infer message types from PCAPs using Canberra dissimilarity as feature to determine segment similarity. Similar segments are then aligned in multiple messages and clustered based on their alignment scores. The clusters denote the inferred message types of similar structure.

* `nemetyl_align-segments.py`

  Run NEMETYL and write the analysis results to a subfolder of `reports/`.



### netzob_*
Infer PCAP with Netzob and compare the result to the tshark dissector of the protocol.

* `netzob_fms.py`
  Run [Netzob](https://github.com/netzob/) and validate it by calculating the FMS for each inferred message.
  Writes the results as text files to a timestamped subfolder of `reports/` 

* `netzob_messagetypes.py`

  Infer PCAP with Netzob and compare the result to the tshark dissector of the protocol.

  


