

# Kache.ai - Annotation Guidelines

The goal of this post is to provide instructions for the new machine labeling policy. These guidelines also serve as a getting started guide to introduce the [Scalabel Platform](http://scalabel.ai), created by the [Berkeley Deep Drive Team](http://bdd-data.berkeley.edu/).


## Scalabel Labeling Web Application

For object detection, Kache uses [Scalabel](http://scalabel.ai), which provides us a variety of advantages over the legacy systems (VGG, etc). For example, [scalabel](http://scalabel.ai) allow us to tag images with custom attributes. In addition to labeling categories (e.g., cars), we can advantageously encode features from the environment such as the **occlusion** and **truncation** parameters. 
In addition, [Scalabel](http://scalabel.ai) provides a flexible and robust data format that supports labeling pipelines particularly suited for autonomous driving. We will be utilizing [Scalabel](http://scalabel.ai) as the premiere tool for providing human annotations moving forward.

---

### Getting Started - Navigating the Web App

To create a project, navigate to the [Scalabel Application](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/) and simply upload the relevant **annotations file** for correction. As an example, I preloaded the BDD100k dataset for visualization:

---

|  BDD100K - PT1    |  BDD100K - PT2   |  BDD100K - PT3    |  BDD100K - PT4    |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|    [PT1](http://ec2-34-220-66-194.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT1_4&task_index=0)   |   [PT2](http://ec2-34-220-66-194.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT2_4&task_index=0)   |   [PT3](ec2-34-220-66-194.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT3_4&task_index=0)   |    [PT4](ec2-34-220-66-194.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT4_4&task_index=0)   |

---

### Getting Started - Creating a New Project & Uploading an Annotations File


* Once you see the screen shown below, click **Create A New Project** 
* Next, choose a project name (No spaces allowed). For bounding box corrections, click on **Image** and **2D Bounding Box** from the dropdown menus. Depending on the project, the next uploads will vary depending on the task, but the scalabel examples provide the **categories.yml** and **bbox_attributes.yml** configuration files.
* Upload your **annotations file** into the Item list (*see below*).
* Select a **Task Size** this size determines the number of chunks you want to break up your annotations file. For example, if you have an item list of **10,000** images and your task size is 10, the you will have generated **10 task lists**; each with **1000** items. The 10 task lists will be assigned to the project name of your choosing.
* The vendor id will be **0** for now, in the future we will use this field to represent different users of a company (for QA purposes). 


|   Project Creation - Scalabel   |

|Homepage |Create Project |Upload Configs |Upload Annotations |
|:-----------------:|:---------------:|:--------------------:|:-------------------:|
|![homepage][image1]|![create][image2]|![upload_cfgs][image3]|![upload_anns][image4]|



Generally, the annotations file will be provided by internal scripts or databse which selects images intended for training nets. Upload the given document into the *item list*.


---

And that's it. Clicking **Enter** will generate a new task list. Clicking on **Go To Project Dashboard** on the next page will give you the option to click one the tasks generated and begin annotating. Also from the dashboard, you will have the option to <font color='red'>export your results into bdd format for training</font> 


![proj_dash][image6]

---


![proj_begin][image7]

---

# Style Guide: 

In this section, we will explore the BDD distibution and define our policy for labeling. 

---

## Style Guides: <font color='green'>Good Techniques</font> 

We would like to continue to maintain BDD's granularity and attribute associations in all new images. this will allow us to easily blend our interal data with the BDD distribution. In particular, we will maintain using occlusion and truncation for all pertaining objects. For our purposes, the respective attributes are defined as follows:

- **Occlusion:** When one object is hidden by another object that passes between it and the observer. The term refers to any situation in which an object in the foreground blocks from view (occults) an object in the background. In our sense, occlusion applies to the visual scene observed from computer-generated imagery when foreground objects obscure distant objects dynamically, as the scene changes over time. 


| Occlusion Examples| Occlusion Examples|
|:-----------------:|:-----------------:|
|![image68][image68]|![image69][image69]|
|![image70][image70]|![image71][image71]|
 
 
- **Truncation:** The bounding box of the object specified does not correspond to the full extent of the object e.g. an image of a person from the waist up, or a view of a car extending outside the outside the field of view of the camera/image, such that it is partially shown. It is assumed with truncated images that a subsequent perspective shift will then clarify the bounds in the view of the truncated object.

|Truncation Examples |Truncation Examples |
|:-----------------:|:-----------------:|
|![][image72]|![][image73]|
|![][image74]|![][image75]|

---

## Style Guides: <font color='red'> Mis-Labels / Mis-Classifications / Class-Ambiguities</font> 

Overall the BDD dataset is remarkably accurate and the class ambiguities are nuanced, however there are some consistent miscategorizations that we would like fixed in our final production dataset. Among those subtleties, a few features we should aim to correct are as follows:

---

Semi-Truck Attribute Labels


Our definition of a truck is somewhat different from the definition in the BDD dataset. To avoid this confusion, we have set up the **Semi** attribute to assign to truck labels of vehicles which **require a class A license or similar to drive.** Examples of trucks which **do not** require a Semi tag are pickups, tow trucks, RVs, trailers, etc. 

Another proper **Semi** truck type are the [**Towing Wrecker Vehicles**](https://www.google.com/search?q=towing+wreckers&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiq7JCEsvfdAhWFJnwKHRfNB-gQ_AUIECgD&biw=1855&bih=990). The example above is a tow truck but should be labeled as a truck without semi for the reasons explained below in the class taxonomy.


Similar to the trucks class, our definition of a bus is more constrained. We will define a bus as only those vehicles which are large enough to have the typical bus-styled swinging door openening mechanism which attached to a turn crank. An example of a truck mis-label is shown above. In this case, we want to classify this vehicle as a bus because it likely has a door that need to swing out to open. As it is a shuttle, it is also likely to make frequent stops.


Car Mis-Labels

---

|  Example 1   |  Example 2 |  Example 3 | Example 4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
| ![car_antenna][image12]| ![car_mislabel0][image15]| ![car_mislabel1][image13]|  ![car_mislabel2][image14] |

---

With the exceptions to the above, our definition of cars is identical to BDD.. One small difference to note is that we will define the car bounding box to include the car itself, excluding protruding items like the car atenna mounted above the car.

In the other examples, the car classes are either a) mis-labeled altogether, or b) labeled, but missing the proper **occlusion** or **truncation** attrubutes. We must clean these instances up.


## Mis-Labeled Examples

---

|Mis-Labeled Examples|
|:--------------:|
|![mislabel1][image76]|
|![mislabel2][image77]|
|![mislabel3][image78]|
|![mislabel4][image79]|
|![mislabel5][image80]|
|![mislabel6][image81]|
|![mislabel7][image82]|
|![mislabel8][image83]|
|![mislabel9][image84]|
|![mislabel10][image85]|
|![mislabel11][image86]|
|![mislabel12][image87]|
|![mislabel13][image88]|
|![mislabel14][image89]|
|![mislabel15][image90]|

---


# Class Taxonomy

Another great feature of Scalabel is that you can easily change the classification template to whatever task suits your need. We intend to leverage these capabilities by customizing our class naming scheme. This guide will serve as the official instructions on which classes to label unless there are special circumstances.  As of now, those classes are:

- person
- rider (person atop a motorized vehicle or bicycle)
- car
- truck(special regard to **Semi** truck attribute)
- bus
- train
- motor
- bike
- traffic sign
- traffic light
- trailer
- construct-cone
- construct-sign
- construct-barrel
- construct-pole
- construct-equipment
- traffic light-red
- traffic light-amber
- traffic light-green
- traffic sign-stop_sign
- traffic sign-slow_sign
- traffic sign-speed_sign


And more to come.

--- 

## Class Taxonomy: Car

This category includes vans, lightweight utility vehicles, SUVs, sedans, hatchbacks, classic cars, sports cars, exotic cars, etc. This category also includes special purpose compact vehicles. i.e., Mini Shuttles, Meter Maids
- Note: Any type of **pickup truck** will be excluded from this category and labeled as trucks.


### Class Taxonomy: Van (Car Sub-Category)


|Van Ex. 1 |Van Ex. 2 |
|:-----------:|:-----------:|
|![van1][van1]|![van2][van2]|
|![van3][van3]|![van4][van4]|


|Real-World Van Examples |Real-World Van Examples |
|:-----------:|:-----------:|
|![van5][van5]|![van6][van6]|
|![van7][van7]|![van8][van8]|

---

### Class Taxonomy: Mini-Shuttle (Car Sub-Category)

|Mini-Shuttle Ex. 1 |Mini-Shuttle Ex. 2 |Mini-Shuttle Ex. 3 |Mini-Shuttle Ex. 4 |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|![minishuttle1][minishuttle1]|![minishuttle2][minishuttle2]|![minishuttle3][minishuttle3]|![minishuttle4][minishuttle4]|


|Real-World Mini-Shuttle Examples |Real-World Mini-Shuttle Examples |
|:--------------:|:--------------:|
|![minishuttle5][minishuttle5]|![minishuttle6][minishuttle6]|
|![minishuttle7][minishuttle7]|![minishuttle8][minishuttle8]|

---

## Class Taxonomy: Truck

This category includes pickup trucks (light to heavy duty), trucks/straight trucks (chassis cab trucks), mail delivery trucks,  recreational vehicles (RVs), Motorized Campers, and Semi-Tractors.


---

### Class Taxonomy: Pickup Truck (Truck Sub-Category)


|Pickup Truck Ex.1 |Pickup Truck Ex.2 |
|:-----------------:|:-----------------:|
|![pickup1][pickup1]|![pickup2][pickup2]|
|![pickup3][pickup3]|![pickup4][pickup4]|


|Real-World Pickup Truck Examples |Real-World Pickup Truck Examples |
|:-------------------:|:---------------:|
|![pickup5][pickup5]|![pickup6][pickup6]|
|![pickup7][pickup7]|![pickup8][pickup8]|
|![pickup9][pickup9]|![pickup10][pickup10]|
|![pickup11][pickup11]|![pickup12][pickup12]|
|![pickup13][pickup13]|![pickup14][pickup14]|
|![pickup15][pickup15]|![pickup16][pickup16]|



### Class Taxonomy: Postal/Delivery-Truck (Truck Sub-Category)


|Postal/Delivery-Truck Ex.1 |Postal/Delivery-Truck Ex.2 |Postal/Delivery-Truck Ex.3 |Postal/Delivery-Truck Ex.4 |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|![postal1][postal1]|![postal2][postal2]|![postal3][postal3]|![postal4][postal4]|


|Real-World Postal/Delivery-Truck Examples  |Real-World Postal/Delivery-Truck Examples  |
|:-----------------:|:-----------------:|
|![postal5][postal5]|![postal6][postal6]|
|![postal7][postal7]|![postal8][postal8]|

---

### Class Taxonomy: Recreational Vehicle a.k.a. "RVs" (Truck Sub-Category)

|RV Ex.1 |RV Ex.2 |
|:---------:|:---------:|
|![rv1][rv1]|![rv2][rv2]|
|![rv3][rv3]|![rv4][rv4]|

|Real-World RV Examples |Real-World RV Examples |
|:---------:|:---------:|
|![rv5][rv5]|![rv6][rv6]|
|![rv7][rv7]|![rv8][rv8]|

---

### Class Taxonomy:  Box-Truck (Truck Sub-Category)

This sub-category includes Box Trucks. This category is tricky and commonly mistaken with the Semi category. However, there are several distinctions. For example, the Chassis Cab Box Truck usually has a wider profile, and mostly have roll-up doors. If in doubt just label it as a truck and add the Semi attribute unless you are absolutely sure.

|Chassis-Cab Box Truck Explanation |
|:-----------------:|
|![chassiscab17][chassiscab17]|


|Box-Truck Ex.1 |Box-Truck Ex.2 |Box-Truck Ex.3 |Box-Truck Ex.4 |
|:--------------:|:-------------:|:-------------:|:------------:|
|![boxtruck1][boxtruck1]|![boxtruck2][boxtruck2]|![boxtruck3][boxtruck3]|![boxtruck4][boxtruck4]|


|Real-World Box-Truck Examples |Real-World Box-Truck Examples |
|:--------------:|:--------------:|
|![boxtruck5][boxtruck5]|![boxtruck6][boxtruck6]|
|![boxtruck7][boxtruck7]|![boxtruck8][boxtruck8]|
|![boxtruck9][boxtruck9]|![boxtruck10][boxtruck10]|
|![boxtruck11][boxtruck11]|![boxtruck12][boxtruck12]|

---

### Class Taxonomy:  Chassis Cab-Truck (Truck Sub-Category)

This sub-category includes Chassis Cab trucks that do not belong to the Box Truck subcategory. 
- Note: Chassis Cab trucks have some trailer or machinery **permanently attached** to the chassis of the truck. Common examples include **garbage trucks, concrete trucks**, and **box trucks**


|Chassis Cab-Truck Ex.1 |Chassis Cab-Truck Ex.2 |Chassis Cab-Truck Ex.3 |Chassis Cab-Truck Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![chassiscab1][chassiscab1]|![chassiscab2][chassiscab2]|![chassiscab3][chassiscab4]|![chassiscab4][chassiscab4]|


|Real-World Chassis Cab-Truck Examples |Real-World Chassis Cab-Truck Examples |
|:--------------:|:--------------:|
|![chassiscab5][chassiscab5]|![chassiscab6][chassiscab6]|
|![chassiscab7][chassiscab7]|![chassiscab8][chassiscab8]|
|![chassiscab9][chassiscab9]|![chassiscab10][chassiscab10]|
|![chassiscab11][chassiscab11]|![chassiscab12][chassiscab12]|

---

### Class Taxonomy: Semi-Truck (CRUCIAL Truck Sub-Category)

Not to be confused with Chasis Cab Trucks (See above), this sub-category corresponds to what are known as Semi-Trucks. A Semi Truck has a fifth wheel coupling to attach various types of semi trailers. For our labeling purposes, these trucks should be labeled as trucks with the semi attribute whether or not the corresponding semi-trailer is attached to it. Of note, semi trucks are mechanically different from the earlier mentioned chassis cab trucks in that:
1. The trailer component is *detachable* from the chassis of the car.
2. The trailer component is *articulated*  Many types of trailers (https://en.wikipedia.org/wiki/Semi-trailer)

- Note: These examples should be labeled as **truck** with the additional **Semi** attribute to distinguish from all other trucks in the category.

|Semi Definition |
|:-----------------:|
|![semi21][semi21]|



---

|Semi-Truck Examples |Semi-Truck Examples |Semi-Truck Examples |Semi-Truck Examples |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![semi1][semi1]|![semi2][semi2]|![semi3][semi3]|![semi4][semi4]|
|![semi5][semi5]|![semi6][semi6]|![semi7][semi7]|![semi8][semi8]|

|Real-World Semi-Truck Examples |Real-World Semi-Truck Examples |
|:--------------:|:--------------:|
|![semi9][semi9]|![semi10][semi10]|
|![semi11][semi11]|![semi12][semi12]|
|![semi13][semi13]|![semi14][semi14]|
|![semi15][semi15]|![semi16][semi16]|
|![semi17][semi17]|![semi18][semi18]|
|![semi19][semi19]|![semi20][semi20]|


---

## Class Taxonomy: Bus

This category includes tour buses, shuttle buses, school buses, articulated buses, trolleys, and more.  Particular shuttle buses are easy to confuse with vans and some trucks. Buses are utilized for 9+ persons, public transport or long distance transport.

---

### Class Taxonomy: Trolley (Bus Sub-Category)

|Trolley Ex.1 |Trolley Ex.2 |Trolley Ex.3 |Trolley Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![trolley1][trolley1]|![trolley2][trolley2]|![trolley3][trolley3]|![trolley4][trolley4]|


|Real-World Trolley Examples |Real-World Trolley Examples |
|:--------------:|:--------------:|
|![trolley5][trolley5]|![trolley6][trolley6]|
|![trolley7][trolley7]|![trolley8][trolley8]|

---

### Class Taxonomy: Shuttle Bus (Bus Sub-Category)

|Shuttle Bus Ex.1 |Shuttle Bus Ex.2 |Shuttle Bus Ex.3 |Shuttle Bus Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![shuttle1][shuttle1]|![shuttle2][shuttle2]|![shuttle3][shuttle3]|![shuttle4][shuttle4]|


|Real-World Shuttle Bus Examples |Real-World Shuttle Bus Examples |
|:--------------:|:--------------:|
|![shuttle5][shuttle5]|![shuttle6][shuttle6]|
|![shuttle7][shuttle7]|![shuttle8][shuttle8]|

---

### Class Taxonomy: Tour Bus (Bus Sub-Category)

|Tour Bus Ex.1 |Tour Bus Ex.2 |Tour Bus Ex.3 |Tour Bus Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![tourbus1][tourbus1]|![tourbus2][tourbus2]|![tourbus3][tourbus3]|![tourbus4][tourbus4]|


|Real-World Tour Bus Examples |Real-world Tour Bus Examples |
|:--------------:|:--------------:|
|![tourbus5][tourbus5]|![tourbus6][tourbus6]|
|![tourbus7][tourbus7]|![tourbus8][tourbus8]|

---

### Class Taxonomy: Public-Transit Bus (Bus Sub-Category)

|Transit Bus Ex.1 |Transit Bus Ex.2 |Transit Bus Ex.3 |Transit Bus Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![transit1][transit1]|![transit2][transit2]|![transit3][transit3]|![transit4][transit4]|


|Real-World Transit Bus Examples |Real-World Transit Bus Examples |
|:--------------:|:--------------:|
|![transit5][transit5]|![transit6][transit6]|
|![transit7][transit7]|![transit8][transit8]|

---

### Class Taxonomy: Long-Distance Bus (Bus Sub-Category)

|Long-Distance Bus Ex.1 |Long-Distance Bus Ex.2 |Long-Distance Bus Ex.3 |Long-Distance Bus Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![longbus1][longbus1]|![longbus2][longbus2]|![longbus3][longbus3]|![longbus4][longbus4]|


|Real-World Long-Distance Bus Examples |Real-World Long-Distance Bus Examples |
|:--------------:|:--------------:|
|![longbus5][longbus5]|![longbus6][longbus6]|
|![longbus7][longbus7]|![longbus8][longbus8]|

---


## Class Taxonomy: Traffic/Control Signs

Sign installed from the state/city authority, usually for information of the driver/cyclist/pedestrian in an everyday traffic scene, e.g. traffic- signs, direction signs - without their poles. No ads/commercial signs. The front side and back side of a sign containing the information. Note that commercial signs attached to buildings become building, attached to poles or standing on their own become billboard


### Class Taxonomy: Construction Sign(s) (construct-sign Category)
This category includes roadway signs in the United States, which have increasingly used symbols rather than words to convey their message. Symbols provide instant communication with roadway users, overcome language barriers, and are becoming standard for traffic control devices throughout the world. 

- Note: The color of construction signs is an important indicator of the information they contain. The use of red on signs is limited to stop, yield, and prohibition signs.A yellow sign conveys a general warning message; green shows permitted traffic movements or directional guidance; fluorescent yellow/green indicates pedestrian crossings and school zones; orange is used for warning and guidance in **roadway work zones**


|Construction Sign Ex.1 |Construction Sign Ex.2 |Construction Sign Ex.3 |Construction Sign Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![constructsign1][constructsign1]|![constructsign2][constructsign2]|![constructsign3][constructsign3]|![constructsign4][constructsign4]|


|Real-World Construction Sign Examples |Real-World Construction Sign Examples |
|:--------------:|:--------------:|
|![constructsign5][constructsign5]|![constructsign6][constructsign6]|
|![constructsign7][constructsign7]|![constructsign8][constructsign8]|

---

### Class Taxonomy: Slow Sign (traffic sign-slow_sign Category)

|Slow Sign Ex.1 |Slow Sign Ex.2 |Slow Sign Ex.3 |Slow Sign Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![slowsign1][slowsign1]|![slowsign2][slowsign2]|![slowsign3][slowsign3]|![slowsign4][slowsign4]|

---

### Class Taxonomy: Speed Sign (traffic sign-speed_sign Category)

|Speed Sign Ex.1 |Speed Sign Ex.2 |Speed Sign Ex.3 |Speed Sign Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![speedsign1][speedsign1]|![speedsign2][speedsign2]|![speedsign3][speedsign3]|![speedsign4][speedsign4]|


|Real-World Speed Sign Examples |Real-World Speed Sign Examples |
|:--------------:|:--------------:|
|![speedsign5][speedsign5]|![speedsign6][speedsign6]|
|![speedsign7][speedsign7]|![speedsign8][speedsign8]|


---

### Class Taxonomy: Stop Sign (traffic sign-stop_sign Category)

|Stop Sign Ex.1 |Stop Sign Ex.2 |Stop Sign Ex.3 |Stop Sign Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![stopsign1][stopsign1]|![stopsign2][stopsign2]|![stopsign3][stopsign3]|![stopsign4][stopsign4]|


|Real-World Stop Sign Examples |Real-World Stop Sign Examples |
|:---------------------:|:---------------------:|
|![stopsign5][stopsign5]|![stopsign6][stopsign6]|
|![stopsign7][stopsign7]|![stopsign8][stopsign8]|


## Class Taxonomy: Trailer

This category includes any non-motorized vehicle generally intended to be pulled by a motorized vehicle used primarily for hauling. Note: This category thoughtfully and purpousfully excludes **semi-trailers** that are attached to Semi-Tractors. 
- Examples: camper trailer, cargo trailer, utility trailer, semi trailer (*only when fully detached from Semi-tractor*)


---

|Trailer Ex.1 |Trailer Ex.2 | 
|:-------------------:|:-------------------:|
|![trailer1][trailer1]|![trailer2][trailer2]|
|![trailer3][trailer3]|![trailer4][trailer4]|
|![trailer5][trailer5]|![trailer6][trailer6]|
|![trailer7][trailer7]|![trailer8][trailer8]|

---

## Class Taxonomy: Construction Equipment

This category includes any machinery that is used in contruction zones and sites.
- Note: Construction Equipment is usually painted yellow or orange according to federal law. In addition, this machinery, if motorized, is not allowed to drive outside the construction zone.


---


|Construction Sign Ex.1 |Construction Sign Ex.2 |Construction Sign Ex.3 |Construction Sign Ex.4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|![constructequip1][constructequip1]|![constructequip2][constructequip2]|![constructequip3][constructequip3]|![constructequip4][constructequip4]|


|Real-World Construction Sign Examples |Real-World Construction Sign Examples |
|:--------------:|:--------------:|
|![constructequip5][constructequip5]|![constructequip6][constructequip6]|
|![constructequip7][constructequip7]|![constructequip8][constructequip8]|



---

## Class Taxonomy: Rider

This category includes non-pedstrians detected on the road. Our definition of **rider** is a human that would use some device to move a distance of at least 1 meter. Includes, riders/drivers of bicycles, motorbikes, scooters, skateboards, horses, roller-blades, wheel-chairs, road cleaning cars, cars without roof. Note that a visible driver of a car with roof can only be seen through the window. 
- Note: Since holes are not labeled, the human is included in the car label.

---


|Rider Ex.1 |Rider Ex.2 |
|:----------------:|:--------------:|
|![rider1][rider1]|![rider2][rider2]|
|![rider3][rider3]|![rider4][rider4]|


|Real-World Rider Examples |Real-World Rider Examples |
|:---------------:|:---------------:|
|![rider5][rider5]|![rider6][rider6]|
|![rider7][rider7]|![rider8][rider8]|
|![rider9][rider9]|![rider10][rider10]|


---



## <font color='green'> Tips & Tricks </font>

1. Objects that are smaller than <font color='red'>7 * 7 pixels</font> can be ignored. The bounding box smaller than <font color='red'>7 * 7 </font> will turn  <font color='grey'>grey</font> and disappear when you finish.
2. Zoom in with your browser to draw the bounding boxes more accurately.
3. Hit **"h"(keyboard)** to hide category label tags on the bounding boxes, and to show them after hitting **"h"(keyboard)** again.
4. The remove operation is <font color='red'>irreversible</font>.
5. If you refresh the page before submission, all previous history  <font color='red'>will not be saved</font>.


# Developer Appendix:

Here is the Json format which is supported by the training pipeline scripts currently in iPython notebooks *See e.g.*:

* **[`darkernet.py`](https://github.com/deanwebb/darknet/blob/master/darkernet.py)** (python wrapper around darknet.py)
* **[`rails-reactor API`](https://gitlab.railsreactor.com/kache-ai/data-pipeline)**
* **`bdd-formatter ros/atm modules`** 
* **[`data_to_coco`](https://github.com/deanwebb/data_to_coco)** 


---

## Data-Format

Our dataser is compatible with the labels generated by [Scalabel](http://www.scalabel.ai/). A label [json](https://google.github.io/styleguide/jsoncstyleguide.xml) file is a list of frame objects with the fields below:

```
- name: string
- url: string
- videoName: string (optional)
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- intrinsics
    - focal: [x, y]
    - center: [x, y]
    - nearClip:
- extrinsics
    - location
    - rotation
- timestamp: int64 (epoch time ms)
- index: int (optional, frame index in this video)
- labels [ ]:
    - id: int32
    - category: string (classification)
    - manual: boolean (whether this label is created or modified manually)
    - attributes:
        - semi: boolean
        - occluded: boolean
        - truncated: boolean
        - trafficLightColor: "red|green|yellow|none"
        - areaType: "direct | alternative" (for driving area)
        - laneDirection: "parallel|vertical" (for lanes)
        - laneStyle: "solid | dashed" (for lanes)
        - laneTypes: (for lanes)
    - box2d:
       - x1: float
       - y1: float
       - x2: float
       - y2: float
   - box3d:
       - alpha: (observation angle if there is a 2D view)
       - orientation: (3D orientation of the bounding box, used for 3D point cloud annotation)
       - location: (3D point, x, y, z, center of the box)
       - dimension: (3D point, height, width, length)
   - poly2d: an array of objects, with the structure
       - vertices: [][]float (list of 2-tuples [x, y])
       - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
       - closed: boolean (closed for polygon and otherwise for path)
```

### BDD100K Details

Road object categories:
```
[
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]
```
They are labeld by `box2d`.

Drivable area category is `drivable area`. There are two area types `areaType`:
```
[
    "alternative",
    "direct"
]
```

Lane marking category is `lane`. There are 8 lane styles `laneStyle`:
```
[
    "crosswalk",
    "double other",
    "double white",
    "double yellow",
    "road curb",
    "single other",
    "single white",
    "single yellow"
]
```

Both drivable areas and lane markings are labeled by `poly2d`. Please check the visulization code [`show_labels.py`](../bdd_data/show_labels.py) for examples of drawing all the labels.

---

[//]: # (Image References)
[image1]: readme_imgs/homepg_resized.png
[image2]: readme_imgs/create_projs_resized.png
[image3]: readme_imgs/upload_cfgs_resized.png
[image4]: readme_imgs/upload_anns_resized.png
[image5]: readme_imgs/post_proj.png
[image6]: readme_imgs/proj_dash.png
[image7]: readme_imgs/export_task_urls.png
[image8]: readme_imgs/truck_v_car1.png
[image9]: readme_imgs/truck_v_car2.png
[image10]: readme_imgs/truck_v_car3.png
[image11]: readme_imgs/bus_v_car.png
[image12]: readme_imgs/car_antenna_resized.png
[image13]: readme_imgs/car_mislabel1_resized.png
[image14]: readme_imgs/car_mislabel2_resized.png
[image15]: readme_imgs/car_mislabel0_resized.png
[van1]: readme_imgs/van1_resized.png
[van2]: readme_imgs/van2_resized.png
[van3]: readme_imgs/van3_resized.png
[van4]: readme_imgs/van4_resized.png
[van5]: readme_imgs/van5_resized.png
[van6]: readme_imgs/van6_resized.png
[van7]: readme_imgs/van7_resized.png
[van8]: readme_imgs/van8_resized.png
[van9]: readme_imgs/van9_resized.png
[van10]: readme_imgs/van10_resized.png
[van11]: readme_imgs/van11_resized.png
[van12]: readme_imgs/van12_resized.png
[minishuttle1]: readme_imgs/minishuttle1_resized.png
[minishuttle2]: readme_imgs/minishuttle2_resized.png
[minishuttle3]: readme_imgs/minishuttle3_resized.png
[minishuttle4]: readme_imgs/minishuttle4_resized.png
[minishuttle5]: readme_imgs/minishuttle5_resized.png
[minishuttle6]: readme_imgs/minishuttle6_resized.png
[minishuttle7]: readme_imgs/minishuttle7_resized.png
[minishuttle8]: readme_imgs/minishuttle8_resized.png
[minishuttle9]: readme_imgs/minishuttle9_resized.png
[minishuttle10]: readme_imgs/minishuttle10_resized.png
[minishuttle11]: readme_imgs/minishuttle11_resized.png
[minishuttle12]: readme_imgs/minishuttle12_resized.png
[pickup1]: readme_imgs/pickup1_resized.png
[pickup2]: readme_imgs/pickup2_resized.png
[pickup3]: readme_imgs/pickup3_resized.png
[pickup4]: readme_imgs/pickup4_resized.png
[pickup5]: readme_imgs/pickup5_resized.png
[pickup6]: readme_imgs/pickup6_resized.png
[pickup7]: readme_imgs/pickup7_resized.png
[pickup8]: readme_imgs/pickup8_resized.png
[pickup9]: readme_imgs/pickup9_resized.png
[pickup10]: readme_imgs/pickup10_resized.png
[pickup11]: readme_imgs/pickup11_resized.png
[pickup12]: readme_imgs/pickup12_resized.png
[pickup13]: readme_imgs/pickup13_resized.png
[pickup14]: readme_imgs/pickup14_resized.png
[pickup15]: readme_imgs/pickup15_resized.png
[pickup16]: readme_imgs/pickup16_resized.png
[postal1]: readme_imgs/postal1_resized.png
[postal2]: readme_imgs/postal2_resized.png
[postal3]: readme_imgs/postal3_resized.png
[postal4]: readme_imgs/postal4_resized.png
[postal5]: readme_imgs/postal5_resized.png
[postal6]: readme_imgs/postal6_resized.png
[postal7]: readme_imgs/postal7_resized.png
[postal8]: readme_imgs/postal8_resized.png
[postal9]: readme_imgs/postal9_resized.png
[postal10]: readme_imgs/postal10_resized.png
[postal11]: readme_imgs/postal11_resized.png
[postal12]: readme_imgs/postal12_resized.png
[postal13]: readme_imgs/postal13_resized.png
[postal14]: readme_imgs/postal14_resized.png
[postal15]: readme_imgs/postal15_resized.png
[postal16]: readme_imgs/postal16_resized.png
[rv1]: readme_imgs/rv1_resized.png
[rv2]: readme_imgs/rv2_resized.png
[rv3]: readme_imgs/rv3_resized.png
[rv4]: readme_imgs/rv4_resized.png
[rv5]: readme_imgs/rv5_resized.png
[rv6]: readme_imgs/rv6_resized.png
[rv7]: readme_imgs/rv7_resized.png
[rv8]: readme_imgs/rv8_resized.png
[rv9]: readme_imgs/rv9_resized.png
[rv10]: readme_imgs/rv10_resized.png
[rv11]: readme_imgs/rv11_resized.png
[rv12]: readme_imgs/rv12_resized.png
[rv13]: readme_imgs/rv13_resized.png
[rv14]: readme_imgs/rv14_resized.png
[rv15]: readme_imgs/rv15_resized.png
[rv16]: readme_imgs/rv16_resized.png
[boxtruck1]: readme_imgs/boxtruck1_resized.png
[boxtruck2]: readme_imgs/boxtruck2_resized.png
[boxtruck3]: readme_imgs/boxtruck3_resized.png
[boxtruck4]: readme_imgs/boxtruck4_resized.png
[boxtruck5]: readme_imgs/boxtruck5_resized.png
[boxtruck6]: readme_imgs/boxtruck6_resized.png
[boxtruck7]: readme_imgs/boxtruck7_resized.png
[boxtruck8]: readme_imgs/boxtruck8_resized.png
[boxtruck9]: readme_imgs/boxtruck9_resized.png
[boxtruck10]: readme_imgs/boxtruck10_resized.png
[boxtruck11]: readme_imgs/boxtruck11_resized.png
[boxtruck12]: readme_imgs/boxtruck12_resized.png
[boxtruck13]: readme_imgs/boxtruck13_resized.png
[boxtruck14]: readme_imgs/boxtruck14_resized.png
[boxtruck15]: readme_imgs/boxtruck15_resized.png
[boxtruck16]: readme_imgs/boxtruck16_resized.png
[chassiscab1]: readme_imgs/chassiscab1_resized.png
[chassiscab2]: readme_imgs/chassiscab2_resized.png
[chassiscab3]: readme_imgs/chassiscab3_resized.png
[chassiscab4]: readme_imgs/chassiscab4_resized.png
[chassiscab5]: readme_imgs/chassiscab5_resized.png
[chassiscab6]: readme_imgs/chassiscab6_resized.png
[chassiscab7]: readme_imgs/chassiscab7_resized.png
[chassiscab8]: readme_imgs/chassiscab8_resized.png
[chassiscab9]: readme_imgs/chassiscab9_resized.png
[chassiscab10]: readme_imgs/chassiscab10_resized.png
[chassiscab11]: readme_imgs/chassiscab11_resized.png
[chassiscab12]: readme_imgs/chassiscab12_resized.png
[chassiscab13]: readme_imgs/chassiscab13_resized.png
[chassiscab14]: readme_imgs/chassiscab14_resized.png
[chassiscab15]: readme_imgs/chassiscab15_resized.png
[chassiscab16]: readme_imgs/chassiscab16_resized.png
[chassiscab17]: readme_imgs/chassiscab17.png
[semi1]: readme_imgs/semi1_resized.png
[semi2]: readme_imgs/semi2_resized.png
[semi3]: readme_imgs/semi3_resized.png
[semi4]: readme_imgs/semi4_resized.png
[semi5]: readme_imgs/semi5_resized.png
[semi6]: readme_imgs/semi6_resized.png
[semi7]: readme_imgs/semi7_resized.png
[semi8]: readme_imgs/semi8_resized.png
[semi9]: readme_imgs/semi9_resized.png
[semi10]: readme_imgs/semi10_resized.png
[semi11]: readme_imgs/semi11_resized.png
[semi12]: readme_imgs/semi12_resized.png
[semi13]: readme_imgs/semi13_resized.png
[semi14]: readme_imgs/semi14_resized.png
[semi15]: readme_imgs/semi15_resized.png
[semi16]: readme_imgs/semi16_resized.png
[semi17]: readme_imgs/semi17_resized.png
[semi18]: readme_imgs/semi18_resized.png
[semi19]: readme_imgs/semi19_resized.png
[semi20]: readme_imgs/semi20_resized.png
[semi21]: readme_imgs/semi21.png
[trolley1]: readme_imgs/trolley1_resized.png
[trolley2]: readme_imgs/trolley2_resized.png
[trolley3]: readme_imgs/trolley3_resized.png
[trolley4]: readme_imgs/trolley4_resized.png
[trolley5]: readme_imgs/trolley5_resized.png
[trolley6]: readme_imgs/trolley6_resized.png
[trolley7]: readme_imgs/trolley7_resized.png
[trolley8]: readme_imgs/trolley8_resized.png
[trolley9]: readme_imgs/trolley9_resized.png
[trolley10]: readme_imgs/trolley10_resized.png
[trolley11]: readme_imgs/trolley11_resized.png
[trolley12]: readme_imgs/trolley12_resized.png
[trolley13]: readme_imgs/trolley13_resized.png
[trolley14]: readme_imgs/trolley14_resized.png
[trolley15]: readme_imgs/trolley15_resized.png
[trolley16]: readme_imgs/trolley16_resized.png
[trolley17]: readme_imgs/trolley17_resized.png
[trolley18]: readme_imgs/trolley18_resized.png
[trolley19]: readme_imgs/trolley19_resized.png
[trolley20]: readme_imgs/trolley20_resized.png
[shuttle1]: readme_imgs/shuttle1_resized.png
[shuttle2]: readme_imgs/shuttle2_resized.png
[shuttle3]: readme_imgs/shuttle3_resized.png
[shuttle4]: readme_imgs/shuttle4_resized.png
[shuttle5]: readme_imgs/shuttle5_resized.png
[shuttle6]: readme_imgs/shuttle6_resized.png
[shuttle7]: readme_imgs/shuttle7_resized.png
[shuttle8]: readme_imgs/shuttle8_resized.png
[shuttle9]: readme_imgs/shuttle9_resized.png
[shuttle10]: readme_imgs/shuttle10_resized.png
[shuttle11]: readme_imgs/shuttle11_resized.png
[shuttle12]: readme_imgs/shuttle12_resized.png
[shuttle13]: readme_imgs/shuttle13_resized.png
[shuttle14]: readme_imgs/shuttle14_resized.png
[shuttle15]: readme_imgs/shuttle15_resized.png
[shuttle16]: readme_imgs/shuttle16_resized.png
[tourbus1]: readme_imgs/tourbus1_resized.png
[tourbus2]: readme_imgs/tourbus2_resized.png
[tourbus3]: readme_imgs/tourbus3_resized.png
[tourbus4]: readme_imgs/tourbus4_resized.png
[tourbus5]: readme_imgs/tourbus5_resized.png
[tourbus6]: readme_imgs/tourbus6_resized.png
[tourbus7]: readme_imgs/tourbus7_resized.png
[tourbus8]: readme_imgs/tourbus8_resized.png
[tourbus9]: readme_imgs/tourbus9_resized.png
[tourbus10]: readme_imgs/tourbus10_resized.png
[tourbus11]: readme_imgs/tourbus11_resized.png
[tourbus12]: readme_imgs/tourbus12_resized.png
[tourbus13]: readme_imgs/tourbus13_resized.png
[tourbus14]: readme_imgs/tourbus14_resized.png
[tourbus15]: readme_imgs/tourbus15_resized.png
[tourbus16]: readme_imgs/tourbus16_resized.png
[transit1]: readme_imgs/transit1_resized.png
[transit2]: readme_imgs/transit2_resized.png
[transit3]: readme_imgs/transit3_resized.png
[transit4]: readme_imgs/transit4_resized.png
[transit5]: readme_imgs/transit5_resized.png
[transit6]: readme_imgs/transit6_resized.png
[transit7]: readme_imgs/transit7_resized.png
[transit8]: readme_imgs/transit8_resized.png
[transit9]: readme_imgs/transit9_resized.png
[transit10]: readme_imgs/transit10_resized.png
[transit11]: readme_imgs/transit11_resized.png
[transit12]: readme_imgs/transit12_resized.png
[transit13]: readme_imgs/transit13_resized.png
[transit14]: readme_imgs/transit14_resized.png
[transit15]: readme_imgs/transit15_resized.png
[transit16]: readme_imgs/transit16_resized.png
[longbus1]: readme_imgs/longbus1_resized.png
[longbus2]: readme_imgs/longbus2_resized.png
[longbus3]: readme_imgs/longbus3_resized.png
[longbus4]: readme_imgs/longbus4_resized.png
[longbus5]: readme_imgs/longbus5_resized.png
[longbus6]: readme_imgs/longbus6_resized.png
[longbus7]: readme_imgs/longbus7_resized.png
[longbus8]: readme_imgs/longbus8_resized.png
[longbus9]: readme_imgs/longbus9_resized.png
[longbus10]: readme_imgs/longbus10_resized.png
[longbus11]: readme_imgs/longbus11_resized.png
[longbus12]: readme_imgs/longbus12_resized.png
[longbus13]: readme_imgs/longbus13_resized.png
[longbus14]: readme_imgs/longbus14_resized.png
[longbus15]: readme_imgs/longbus15_resized.png
[longbus16]: readme_imgs/longbus16_resized.png
[constructsign1]: readme_imgs/constructsignn1_resized.png
[constructsign2]: readme_imgs/constructsign2_resized.png
[constructsign3]: readme_imgs/constructsignn3_resized.png
[constructsign4]: readme_imgs/constructsign4_resized.png
[constructsign5]: readme_imgs/constructsign5_resized.png
[constructsign6]: readme_imgs/constructsign6_resized.png
[constructsign7]: readme_imgs/constructsign7_resized.png
[constructsign8]: readme_imgs/constructsign8_resized.png
[constructsign9]: readme_imgs/constructsign9_resized.png
[constructsign10]: readme_imgs/constructsign10_resized.png
[constructsign11]: readme_imgs/constructsign11_resized.png
[constructsign12]: readme_imgs/constructsign12_resized.png
[constructsign13]: readme_imgs/constructsign13_resized.png
[constructsign14]: readme_imgs/constructsign14_resized.png
[constructsign15]: readme_imgs/constructsign15_resized.png
[constructsign16]: readme_imgs/constructsign16_resized.png
[slowsign1]: readme_imgs/slowsign1_resized.png
[slowsign2]: readme_imgs/slowsign2_resized.png
[slowsign3]: readme_imgs/slowsign3_resized.png
[slowsign4]: readme_imgs/slowsign4_resized.png
[slowsign5]: readme_imgs/slowsign5_resized.png
[slowsign6]: readme_imgs/slowsign6_resized.png
[slowsign7]: readme_imgs/slowsign7_resized.png
[slowsign8]: readme_imgs/slowsign8_resized.png
[slowsign9]: readme_imgs/slowsign9_resized.png
[slowsign10]: readme_imgs/slowsign10_resized.png
[slowsign11]: readme_imgs/slowsign11_resized.png
[slowsign12]: readme_imgs/slowsign12_resized.png
[slowsign13]: readme_imgs/slowsign13_resized.png
[slowsign14]: readme_imgs/slowsign14_resized.png
[slowsign15]: readme_imgs/slowsign15_resized.png
[slowsign16]: readme_imgs/slowsign16_resized.png
[speedsign1]: readme_imgs/speedsign1_resized.png
[speedsign2]: readme_imgs/speedsign2_resized.png
[speedsign3]: readme_imgs/speedsign3_resized.png
[speedsign4]: readme_imgs/speedsign4_resized.png
[speedsign5]: readme_imgs/speedsign5_resized.png
[speedsign6]: readme_imgs/speedsign6_resized.png
[speedsign7]: readme_imgs/speedsign7_resized.png
[speedsign8]: readme_imgs/speedsign8_resized.png
[speedsign9]: readme_imgs/speedsign9_resized.png
[speedsign10]: readme_imgs/speedsign10_resized.png
[speedsign11]: readme_imgs/speedsign11_resized.png
[speedsign12]: readme_imgs/speedsign12_resized.png
[speedsign13]: readme_imgs/speedsign13_resized.png
[speedsign14]: readme_imgs/speedsign14_resized.png
[speedsign15]: readme_imgs/speedsign15_resized.png
[speedsign16]: readme_imgs/speedsign16_resized.png
[stopsign1]: readme_imgs/stopsign1_resized.png
[stopsign2]: readme_imgs/stopsign2_resized.png
[stopsign3]: readme_imgs/stopsign3_resized.png
[stopsign4]: readme_imgs/stopsign4_resized.png
[stopsign5]: readme_imgs/stopsign5_resized.png
[stopsign6]: readme_imgs/stopsign6_resized.png
[stopsign7]: readme_imgs/stopsign7_resized.png
[stopsign8]: readme_imgs/stopsign8_resized.png
[stopsign9]: readme_imgs/stopsign9_resized.png
[stopsign10]: readme_imgs/stopsign10_resized.png
[stopsign11]: readme_imgs/stopsign11_resized.png
[stopsign12]: readme_imgs/stopsign12_resized.png
[stopsign13]: readme_imgs/stopsign13_resized.png
[stopsign14]: readme_imgs/stopsign14_resized.png
[stopsign15]: readme_imgs/stopsign15_resized.png
[stopsign16]: readme_imgs/stopsign16_resized.png
[trailer1]: readme_imgs/trailer1_resized.png
[trailer2]: readme_imgs/trailer2_resized.png
[trailer3]: readme_imgs/trailer3_resized.png
[trailer4]: readme_imgs/trailer4_resized.png
[trailer5]: readme_imgs/trailer5_resized.png
[trailer6]: readme_imgs/trailer6_resized.png
[trailer7]: readme_imgs/trailer7_resized.png
[trailer8]: readme_imgs/trailer8_resized.png
[trailer9]: readme_imgs/trailer9_resized.png
[trailer10]: readme_imgs/trailer10_resized.png
[trailer11]: readme_imgs/trailer11_resized.png
[trailer12]: readme_imgs/trailer12_resized.png
[trailer13]: readme_imgs/trailer13_resized.png
[trailer14]: readme_imgs/trailer14_resized.png
[trailer15]: readme_imgs/trailer15_resized.png
[trailer16]: readme_imgs/trailer16_resized.png
[constructequip1]: readme_imgs/constructequip1_resized.png
[constructequip2]: readme_imgs/constructequip2_resized.png
[constructequip3]: readme_imgs/constructequip3_resized.png
[constructequip4]: readme_imgs/constructequip4_resized.png
[constructequip5]: readme_imgs/constructequip5_resized.png
[constructequip6]: readme_imgs/constructequip6_resized.png
[constructequip7]: readme_imgs/constructequip7_resized.png
[constructequip8]: readme_imgs/constructequip8_resized.png
[constructequip9]: readme_imgs/constructequip9_resized.png
[constructequip10]: readme_imgs/constructequip10_resized.png
[constructequip11]: readme_imgs/constructequip11_resized.png
[constructequip12]: readme_imgs/constructequip12_resized.png
[constructequip13]: readme_imgs/constructequip13_resized.png
[constructequip14]: readme_imgs/constructequip14_resized.png
[constructequip15]: readme_imgs/constructequip15_resized.png
[constructequip16]: readme_imgs/constructequip16_resized.png
[rider1]: readme_imgs/rider1_resized.png
[rider2]: readme_imgs/rider2_resized.png
[rider3]: readme_imgs/rider3_resized.png
[rider4]: readme_imgs/rider4_resized.png
[rider5]: readme_imgs/rider5_resized.png
[rider6]: readme_imgs/rider6_resized.png
[rider7]: readme_imgs/rider7_resized.png
[rider8]: readme_imgs/rider8_resized.png
[rider9]: readme_imgs/rider9_resized.png
[rider10]: readme_imgs/rider10_resized.png
[rider11]: readme_imgs/rider11_resized.png
[rider12]: readme_imgs/rider12_resized.png
[rider13]: readme_imgs/rider13_resized.png
[rider14]: readme_imgs/rider14_resized.png
[rider15]: readme_imgs/rider15_resized.png
[rider16]: readme_imgs/rider16_resized.png
[image68]: readme_imgs/occlusion1_resized.png
[image69]: readme_imgs/occlusion2_resized.png
[image70]: readme_imgs/occlusion3_resized.png
[image71]: readme_imgs/occlusion4_resized.png
[image72]: readme_imgs/truncation1_resized.png
[image73]: readme_imgs/truncation2_resized.png
[image74]: readme_imgs/truncation3_resized.png
[image75]: readme_imgs/truncation4_resized.png
[image76]: readme_imgs/label_qa_slide2.png
[image77]: readme_imgs/label_qa_slide3.png
[image78]: readme_imgs/label_qa_slide4.png
[image79]: readme_imgs/label_qa_slide5.png
[image80]: readme_imgs/label_qa_slide6.png
[image81]: readme_imgs/label_qa_slide7.png
[image82]: readme_imgs/label_qa_slide8.png
[image83]: readme_imgs/label_qa_slide9.png
[image84]: readme_imgs/label_qa_slide10.png
[image85]: readme_imgs/label_qa_slide11.png
[image86]: readme_imgs/label_qa_slide12.png
[image87]: readme_imgs/label_qa_slide13.png
[image88]: readme_imgs/label_qa_slide14.png
[image89]: readme_imgs/label_qa_slide15.png
[image90]: readme_imgs/label_qa_slide16.png


```python

```
