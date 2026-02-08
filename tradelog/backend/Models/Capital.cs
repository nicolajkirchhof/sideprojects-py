namespace backend.net.Models;

public class Capital
{
    public int Id { get; set; }
    public DateTime Date { get; set; }
    public float NetLiquidity { get; set; }
    public float ExcessLiquidity { get; set; }
    public float Bpr { get; set; }
}
