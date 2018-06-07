import java.io.*;
import java.util.*;

public class test {

	public static HashMap<String, String> map = new HashMap<String, String>();

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		File file = new File("/Users/tringuyen/Desktop/name.txt");

		ArrayList<String> list2 = new ArrayList<String>();
		ArrayList<String> list = new ArrayList<String>();
		HashMap<Integer, String> nameMap = new HashMap<Integer, String>();
		HashMap<Integer, String> descriptionMap = new HashMap<Integer, String>();

		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Boolean bool = false;

		int a=0;
		String doubleArr[][] = new String[1723][10];

		ArrayList<String> listSplit = new ArrayList<String>();
/*
 * read file and put it in arraylist and hashmap
 */
		while((st=br.readLine())!=null){

			if(st!=null&& bool==false){
				String[] arr1 = st.split("-");
					String word= arr1[0];
					list.add(word);

				bool = true;
			}
			else if(st!=null && bool==true){
				list2.add(st);
				bool = false;
			}
		}
		
		
		for(int i=0; i<list2.size(); i++){
			map.put(list2.get(i), list.get(i));
			nameMap.put(i, list.get(i));
			descriptionMap.put(i, list2.get(i));
		}

		for(int x=0; x<list2.size();x++){
			//for(int y=0; y<doubleArr[x].length;y++){
			if(doubleArr[x].length>=6){
				map.put(doubleArr[x][0], doubleArr[x][7]);
			}else {
				continue;
			}
		}
		
/*
 * get key that have similar values		
 */
		
		Map<String, ArrayList<String>> reverseMap = new HashMap<>();

		for (Map.Entry<String,String> entry : map.entrySet()) {
		    if (!reverseMap.containsKey(entry.getValue())) {
		        reverseMap.put(entry.getValue(), new ArrayList<>());
		    }
		    ArrayList<String> keys = reverseMap.get(entry.getValue());
		    keys.add(entry.getKey());
		    reverseMap.put(entry.getValue(), keys);
		}
		//System.out.println(reverseMap.get("AC"));

/*
 * listsplit represent a list of split string
 * count word repetition
 */
		String arr[];
		for(int x=0; x<list2.size(); x++){
			arr = list2.get(x).split("\\s*[^a-zA-Z]+");
			for(int i=0; i<arr.length;i++){
				listSplit.add(arr[i]);
			}
		}
/*
 * count word repetition
 */
		Set<String> unique = new HashSet<String>(listSplit);
		Map<String, Integer> countMap = new HashMap<String, Integer>();
		for (String key : unique) {
			int accurNO = Collections.frequency(listSplit, key);
			countMap.put(key, accurNO);
			//System.out.println(key + ": " + Collections.frequency(listSplit, key));
		}
		System.out.println("listSplit"+ listSplit);
		Set<String> unique1 = new HashSet<String>(listSplit);
		Map<String, Integer> countMap1 = new HashMap<String, Integer>();
		for (String key : unique) {
			int accurNO = Collections.frequency(listSplit, key);
			countMap1.put(key, accurNO);
			//System.out.println(key + ": " + Collections.frequency(listSplit, key));
		}
		
/*
 * comparing each string to the rating system		
 */
		for(int i=0; i<countMap.size(); i++){
			if(descriptionMap.get(i) != null){
				for(int x=0; x<listSplit.size();x++){
					
				}
			}
		}
		
		
		//System.out.println(list.size()+ " " + list2.size());
		//System.out.println(listSplit.toString());
		System.out.println(countMap.toString());
		//System.out.println(list2.toString()+"\n");
		//System.out.println(map.entrySet());
	}

}
