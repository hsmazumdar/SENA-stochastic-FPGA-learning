module class1_tree14(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[266])) | (i[47] & (~i[141]) & i[349] & i[266]) | (i[107] & (~i[48]) & i[141] & i[349] & i[266]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
