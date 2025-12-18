
public class BinaryTree {
    
    public static void main(String[] args){
        Node root = new Node(8);
        root.add(4);
        root.add(9);
        root.add(1);
        root.add(6);
        root.add(5);
        root.add(7);
        
        printInOrder(root);
        System.out.println();
        printPreOrder(root);
        System.out.println();
        printPostOrder(root);
		
		root.printDFS();
        
        root.printBFS(root);
    }
    
    public static void printInOrder(Node node) {
        if (node != null) {
            printInOrder(node.left);
            System.out.print(" " + node.value);
            printInOrder(node.right);
        }
    }
	
    public static void printPreOrder(Node node) {
        if (node != null) {
            System.out.print(" " + node.value);
            printPreOrder(node.left);
            printPreOrder(node.right);
        }
    }
	
    public static void printPostOrder(Node node) {
        if (node != null) {
            printPostOrder(node.left);
            printPostOrder(node.right);
            System.out.print(" " + node.value);
        }
    }
}
