This is a collection of short scripts relating to video processing and computer vision on Digital Agriculture. 

The goal is to scan damaged corn leaves with the help of a real world tool and diagnose defoliation (percentage missing area).

Our design uses a simple tool, a rectangle built with bright orange rulers (easily detectable color in a corn field) and a backpiece.

The tool design offers the following adavantages:
(1) By sliding the damaged corn leaf between the rulers and the backpiece we can straighten the leaf, avoiding leaf curling and similar issues
(2) The backpiece separates the leaf from background elements (other leaves or corn plants)
(3) The rectangle constructed with the rulers is a known size, as such it can be used as a size reference to derive leaf area

Our "Scan Methodlogy" works the following way:
(1) Record a video of a user sliding the damaged corn leaf between the rulers and backpiece of the tool
(2) Track the bright orange rectangle and crop to that section in the video
(3) For every unique section of the leaf, count the number of leaf pixels and backpiece pixels
(4) Use the known size of the target rectangle to translate pixels to area and derive the leaf area of each section
(5) Aggregate the derived leaf areas of each unique leaf section to calculate the total area of the damaged leaf
