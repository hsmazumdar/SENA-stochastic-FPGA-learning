module myreg #(parameter DataWidth=16) (input wire clk, input [DataWidth-1:0] data_in, output reg [DataWidth-1:0] data_out);

always@(posedge clk) begin
	data_out <= data_in;
end

endmodule
