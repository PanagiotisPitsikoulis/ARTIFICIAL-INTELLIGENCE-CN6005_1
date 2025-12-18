
public class Node {
    int value;
    Node left;
    Node right;
    
    public Node(int value){
        this.value = value;
        this.left = null;
        this.right = null;
    }
    
    public void add(int value){
        if(value==this.value)
            return;
        else if(value<this.value){
            if(left==null)
                left = new Node(value);
            else
                left.add(value);
        }
        else if(value>this.value){
            if(right==null)
                right = new Node(value);
            else
                right.add(value);
        }
    }
	
	public void printDFS(){
        System.out.println(value);
        if(left!=null)
            left.printDFS();
        if(right!=null)
            right.printDFS();
    }
    
    public void printBFS(Node root) {
        if (root == null) return;
        // A queue is used to keep track of nodes at each level. We enqueue the root node first and then, for every node we dequeue, we enqueue its left and right children (if they exist). The queue ensures that nodes are processed level by level.
        ArrayList<Node> queue = new ArrayList<>();
        // Start at the root of the tree, and enqueue the root node
        queue.add(root);
        while (!queue.isEmpty()) {
            Node currentNode = queue.get(0);  // Dequeue the front node
            queue.remove(0);
            System.out.print(currentNode.value + " ");  // Process the current node
            // Enqueue left child of the current node if it exists
            if (currentNode.left != null) {
                queue.add(currentNode.left);
            }
            // Enqueue right child of the current node if it exists
            if (currentNode.right != null) {
                queue.add(currentNode.right);
            }
        }
    }
}
